#!/usr/bin/env python3
"""
AGI Trading System - Mesh Network Coordinator
Implements peer-to-peer communication, fault tolerance, and consensus mechanisms
for direct agent-to-agent communication in trading decisions.
"""

import asyncio
import time
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid


class MessagePriority(Enum):
    """Message priority levels for trading communications"""
    EMERGENCY = 0       # System-wide alerts, emergency stops
    URGENT = 1         # Trading signals requiring immediate action
    STANDARD = 2       # Routine data sharing, market updates
    LOW = 3           # Learning updates, optimization data


class ChannelType(Enum):
    """Communication channel types"""
    EMERGENCY_BROADCAST = "emergency_broadcast"
    TRADING_SIGNALS = "trading_signals"
    MARKET_DATA = "market_data"
    NEURAL_PATTERNS = "neural_patterns"
    SYSTEM_COORDINATION = "system_coordination"


class NodeStatus(Enum):
    """Node status in mesh network"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class NetworkMessage:
    """Network message structure for peer-to-peer communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""  # Empty for broadcast
    channel: ChannelType = ChannelType.SYSTEM_COORDINATION
    priority: MessagePriority = MessagePriority.STANDARD
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ttl: int = 30  # Time to live in seconds
    signature: str = ""
    requires_consensus: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'channel': self.channel.value,
            'priority': self.priority.value,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'ttl': self.ttl,
            'signature': self.signature,
            'requires_consensus': self.requires_consensus
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkMessage':
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            sender_id=data.get('sender_id', ''),
            receiver_id=data.get('receiver_id', ''),
            channel=ChannelType(data.get('channel', 'system_coordination')),
            priority=MessagePriority(data.get('priority', 2)),
            payload=data.get('payload', {}),
            timestamp=data.get('timestamp', time.time()),
            ttl=data.get('ttl', 30),
            signature=data.get('signature', ''),
            requires_consensus=data.get('requires_consensus', False)
        )


@dataclass
class NodeInfo:
    """Information about network nodes"""
    node_id: str
    node_type: str  # Agent type (AGENT_01, AGENT_02, etc.)
    capabilities: List[str]
    status: NodeStatus = NodeStatus.ACTIVE
    last_heartbeat: float = field(default_factory=time.time)
    load_metrics: Dict[str, float] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    reputation_score: float = 1.0


@dataclass
class ConsensusProposal:
    """Consensus proposal for trading decisions"""
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    proposer: str = ""
    proposal_type: str = ""  # trade_decision, risk_adjustment, system_change
    proposal_data: Dict[str, Any] = field(default_factory=dict)
    votes: Dict[str, bool] = field(default_factory=dict)  # node_id -> vote
    threshold: float = 0.67  # 67% consensus required
    timeout: float = field(default_factory=lambda: time.time() + 10)
    status: str = "pending"  # pending, approved, rejected, expired


class MeshNetworkCoordinator:
    """
    Peer-to-peer mesh network coordinator for AGI Trading System
    Implements Byzantine fault tolerance, load balancing, and consensus mechanisms
    """
    
    def __init__(self, node_id: str, node_type: str):
        self.node_id = node_id
        self.node_type = node_type
        self.logger = logging.getLogger(f"MeshNetwork.{node_id}")
        
        # Network state
        self.nodes: Dict[str, NodeInfo] = {}
        self.connections: Dict[str, asyncio.Queue] = {}
        self.message_handlers: Dict[ChannelType, List[Callable]] = {}
        self.consensus_proposals: Dict[str, ConsensusProposal] = {}
        
        # Communication channels
        self.message_queues: Dict[MessagePriority, asyncio.PriorityQueue] = {
            priority: asyncio.PriorityQueue() for priority in MessagePriority
        }
        
        # Network topology configuration
        self.core_agents = ["AGENT_01", "AGENT_02", "AGENT_03", "AGENT_04"]
        self.data_agents = ["AGENT_06", "AGENT_07", "AGENT_08", "AGENT_10"]
        self.ml_agents = ["AGENT_05"]
        self.system_agents = ["AGENT_09", "AGENT_11", "AGENT_12"]
        
        # Fault tolerance
        self.heartbeat_interval = 3.0  # seconds
        self.failure_threshold = 10.0  # seconds
        self.recovery_timeout = 30.0  # seconds
        
        # Performance tracking
        self.message_stats = {
            'sent': 0,
            'received': 0,
            'consensus_decisions': 0,
            'failed_deliveries': 0
        }
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.tasks = []
        
        # Initialize self as a node
        self.nodes[self.node_id] = NodeInfo(
            node_id=self.node_id,
            node_type=self.node_type,
            capabilities=self._get_node_capabilities()
        )
        
        self.logger.info(f"🌐 Mesh network coordinator initialized for {node_id}")
    
    def _get_node_capabilities(self) -> List[str]:
        """Get capabilities for this node type"""
        capability_map = {
            "AGENT_01": ["mt5_connection", "market_access", "trade_execution"],
            "AGENT_02": ["signal_coordination", "decision_aggregation"],
            "AGENT_03": ["risk_calculation", "position_sizing", "drawdown_control"],
            "AGENT_04": ["chart_analysis", "pattern_recognition"],
            "AGENT_05": ["neural_processing", "pattern_learning", "prediction"],
            "AGENT_06": ["technical_analysis", "indicator_calculation"],
            "AGENT_07": ["news_analysis", "sentiment_processing"],
            "AGENT_08": ["optimization", "parameter_tuning"],
            "AGENT_09": ["user_interface", "visualization"],
            "AGENT_10": ["data_management", "streaming"],
            "AGENT_11": ["system_monitoring", "performance_tracking"],
            "AGENT_12": ["signal_validation", "quality_control"]
        }
        return capability_map.get(self.node_type, ["general"])
    
    async def start_network(self):
        """Start the mesh network coordinator"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("🚀 Starting mesh network coordinator...")
        
        # Start core network tasks
        self.tasks = [
            asyncio.create_task(self._heartbeat_sender()),
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._consensus_monitor()),
            asyncio.create_task(self._load_balancer()),
            asyncio.create_task(self._failure_detector())
        ]
        
        self.logger.info("✅ Mesh network coordinator started")
    
    async def stop_network(self):
        """Stop the mesh network coordinator"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("🛑 Stopping mesh network coordinator...")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.logger.info("✅ Mesh network coordinator stopped")
    
    def register_message_handler(self, channel: ChannelType, handler: Callable):
        """Register a message handler for a specific channel"""
        if channel not in self.message_handlers:
            self.message_handlers[channel] = []
        self.message_handlers[channel].append(handler)
        self.logger.info(f"📝 Registered handler for {channel.value} channel")
    
    async def join_network(self, seed_nodes: List[str] = None):
        """Join the mesh network via seed nodes"""
        self.logger.info("🔗 Joining mesh network...")
        
        if seed_nodes:
            for seed_node in seed_nodes:
                await self._establish_connection(seed_node)
        
        # Configure topology based on node type
        await self._configure_network_topology()
        
        self.logger.info(f"✅ Joined mesh network with {len(self.connections)} connections")
    
    async def _configure_network_topology(self):
        """Configure network topology based on trading system requirements"""
        # Full mesh for core trading agents
        if self.node_type in self.core_agents:
            for agent_id in self.core_agents:
                if agent_id != self.node_id:
                    await self._establish_connection(agent_id)
        
        # Hub-and-spoke for data agents to core agents
        elif self.node_type in self.data_agents:
            for core_agent in self.core_agents:
                await self._establish_connection(core_agent)
        
        # Connect ML agent to core and data agents
        elif self.node_type in self.ml_agents:
            for agent_id in self.core_agents + self.data_agents:
                await self._establish_connection(agent_id)
        
        # System agents connect to all others
        elif self.node_type in self.system_agents:
            all_agents = self.core_agents + self.data_agents + self.ml_agents
            for agent_id in all_agents:
                if agent_id != self.node_id:
                    await self._establish_connection(agent_id)
    
    async def _establish_connection(self, peer_id: str):
        """Establish connection with a peer node"""
        if peer_id not in self.connections:
            self.connections[peer_id] = asyncio.Queue(maxsize=1000)
            self.logger.info(f"🔗 Established connection with {peer_id}")
    
    async def send_message(self, message: NetworkMessage) -> bool:
        """Send message to peer(s)"""
        try:
            # Sign the message
            message.signature = self._sign_message(message)
            
            # Broadcast or direct send
            if not message.receiver_id:
                await self._broadcast_message(message)
            else:
                await self._send_direct_message(message)
            
            self.message_stats['sent'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send message: {e}")
            self.message_stats['failed_deliveries'] += 1
            return False
    
    async def _broadcast_message(self, message: NetworkMessage):
        """Broadcast message to all connected peers"""
        for peer_id, queue in self.connections.items():
            try:
                # Add to appropriate priority queue
                await queue.put((message.priority.value, message))
                self.logger.debug(f"📤 Broadcasted {message.channel.value} to {peer_id}")
            except asyncio.QueueFull:
                self.logger.warning(f"⚠️ Queue full for {peer_id}, dropping message")
    
    async def _send_direct_message(self, message: NetworkMessage):
        """Send message directly to specific peer"""
        if message.receiver_id in self.connections:
            try:
                queue = self.connections[message.receiver_id]
                await queue.put((message.priority.value, message))
                self.logger.debug(f"📤 Sent {message.channel.value} to {message.receiver_id}")
            except asyncio.QueueFull:
                self.logger.warning(f"⚠️ Queue full for {message.receiver_id}")
        else:
            self.logger.warning(f"⚠️ No connection to {message.receiver_id}")
    
    async def _message_processor(self):
        """Process incoming messages by priority"""
        while self.running:
            try:
                # Process messages in priority order
                for priority in MessagePriority:
                    try:
                        # Check each connection for messages of this priority
                        for peer_id, queue in self.connections.items():
                            try:
                                priority_val, message = await asyncio.wait_for(
                                    queue.get(), timeout=0.1
                                )
                                
                                if priority_val == priority.value:
                                    await self._handle_message(message)
                                    self.message_stats['received'] += 1
                                else:
                                    # Put back if wrong priority
                                    await queue.put((priority_val, message))
                                    
                            except asyncio.TimeoutError:
                                continue
                            except Exception as e:
                                self.logger.error(f"❌ Error processing message: {e}")
                                
                    except Exception as e:
                        self.logger.error(f"❌ Error in priority processing: {e}")
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"❌ Error in message processor: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: NetworkMessage):
        """Handle incoming message"""
        try:
            # Verify message signature
            if not self._verify_message(message):
                self.logger.warning(f"⚠️ Invalid signature from {message.sender_id}")
                return
            
            # Check TTL
            if time.time() - message.timestamp > message.ttl:
                self.logger.debug(f"⏰ Message expired from {message.sender_id}")
                return
            
            # Handle consensus proposals
            if message.requires_consensus:
                await self._handle_consensus_message(message)
                return
            
            # Route to appropriate handlers
            if message.channel in self.message_handlers:
                for handler in self.message_handlers[message.channel]:
                    try:
                        await handler(message)
                    except Exception as e:
                        self.logger.error(f"❌ Handler error: {e}")
            
            self.logger.debug(f"📥 Processed {message.channel.value} from {message.sender_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error handling message: {e}")
    
    async def _handle_consensus_message(self, message: NetworkMessage):
        """Handle consensus-related messages"""
        payload = message.payload
        
        if payload.get('type') == 'proposal':
            # New consensus proposal
            proposal = ConsensusProposal(
                proposal_id=payload['proposal_id'],
                proposer=message.sender_id,
                proposal_type=payload['proposal_type'],
                proposal_data=payload['proposal_data'],
                threshold=payload.get('threshold', 0.67),
                timeout=time.time() + payload.get('timeout', 10)
            )
            self.consensus_proposals[proposal.proposal_id] = proposal
            
            # Auto-vote based on node capabilities and proposal type
            vote = await self._evaluate_proposal(proposal)
            await self._cast_vote(proposal.proposal_id, vote)
            
        elif payload.get('type') == 'vote':
            # Vote on existing proposal
            proposal_id = payload['proposal_id']
            if proposal_id in self.consensus_proposals:
                proposal = self.consensus_proposals[proposal_id]
                proposal.votes[message.sender_id] = payload['vote']
                
                # Check if consensus reached
                await self._check_consensus(proposal)
    
    async def _evaluate_proposal(self, proposal: ConsensusProposal) -> bool:
        """Evaluate proposal and return vote"""
        # Trading decision evaluation logic
        if proposal.proposal_type == "trade_decision":
            return await self._evaluate_trade_proposal(proposal)
        elif proposal.proposal_type == "risk_adjustment":
            return await self._evaluate_risk_proposal(proposal)
        elif proposal.proposal_type == "system_change":
            return await self._evaluate_system_proposal(proposal)
        else:
            return True  # Default approve for unknown types
    
    async def _evaluate_trade_proposal(self, proposal: ConsensusProposal) -> bool:
        """Evaluate trading decision proposal"""
        data = proposal.proposal_data
        
        # Basic risk checks
        if data.get('risk_percentage', 0) > 0.05:  # > 5% risk
            return False
        
        if data.get('confidence_score', 0) < 0.6:  # < 60% confidence
            return False
        
        # Check if signal is recent
        if time.time() - data.get('signal_timestamp', 0) > 30:  # > 30 seconds old
            return False
        
        return True
    
    async def _evaluate_risk_proposal(self, proposal: ConsensusProposal) -> bool:
        """Evaluate risk adjustment proposal"""
        data = proposal.proposal_data
        
        # Conservative risk management
        if data.get('new_risk_level', 0) > data.get('current_risk_level', 0):
            return False  # Don't approve risk increases
        
        return True
    
    async def _evaluate_system_proposal(self, proposal: ConsensusProposal) -> bool:
        """Evaluate system change proposal"""
        # System agents have more authority on system changes
        if self.node_type in self.system_agents:
            return True
        
        return False  # Non-system agents decline system changes
    
    async def _cast_vote(self, proposal_id: str, vote: bool):
        """Cast vote on consensus proposal"""
        vote_message = NetworkMessage(
            sender_id=self.node_id,
            channel=ChannelType.SYSTEM_COORDINATION,
            priority=MessagePriority.URGENT,
            payload={
                'type': 'vote',
                'proposal_id': proposal_id,
                'vote': vote
            },
            requires_consensus=False
        )
        
        await self.send_message(vote_message)
    
    async def _check_consensus(self, proposal: ConsensusProposal):
        """Check if consensus has been reached"""
        if time.time() > proposal.timeout:
            proposal.status = "expired"
            return
        
        total_votes = len(proposal.votes)
        yes_votes = sum(1 for vote in proposal.votes.values() if vote)
        
        if total_votes >= len(self.nodes) * proposal.threshold:
            if yes_votes / total_votes >= proposal.threshold:
                proposal.status = "approved"
                await self._execute_consensus_decision(proposal)
            else:
                proposal.status = "rejected"
            
            self.message_stats['consensus_decisions'] += 1
    
    async def _execute_consensus_decision(self, proposal: ConsensusProposal):
        """Execute approved consensus decision"""
        self.logger.info(f"✅ Consensus reached for {proposal.proposal_type}")
        
        # Execute based on proposal type
        if proposal.proposal_type == "trade_decision":
            await self._execute_trade_decision(proposal.proposal_data)
        elif proposal.proposal_type == "risk_adjustment":
            await self._execute_risk_adjustment(proposal.proposal_data)
        elif proposal.proposal_type == "system_change":
            await self._execute_system_change(proposal.proposal_data)
    
    async def _execute_trade_decision(self, data: Dict[str, Any]):
        """Execute approved trade decision"""
        # Send execution signal to MT5 connector
        execution_message = NetworkMessage(
            sender_id=self.node_id,
            receiver_id="AGENT_01",  # MT5 Connector
            channel=ChannelType.TRADING_SIGNALS,
            priority=MessagePriority.URGENT,
            payload={
                'action': 'execute_trade',
                'trade_data': data,
                'consensus_approved': True
            }
        )
        
        await self.send_message(execution_message)
    
    async def _execute_risk_adjustment(self, data: Dict[str, Any]):
        """Execute approved risk adjustment"""
        # Send to risk calculator
        risk_message = NetworkMessage(
            sender_id=self.node_id,
            receiver_id="AGENT_03",  # Risk Calculator
            channel=ChannelType.SYSTEM_COORDINATION,
            priority=MessagePriority.URGENT,
            payload={
                'action': 'adjust_risk_parameters',
                'risk_data': data,
                'consensus_approved': True
            }
        )
        
        await self.send_message(risk_message)
    
    async def _execute_system_change(self, data: Dict[str, Any]):
        """Execute approved system change"""
        # Broadcast system change to all nodes
        system_message = NetworkMessage(
            sender_id=self.node_id,
            channel=ChannelType.SYSTEM_COORDINATION,
            priority=MessagePriority.URGENT,
            payload={
                'action': 'system_change',
                'change_data': data,
                'consensus_approved': True
            }
        )
        
        await self.send_message(system_message)
    
    async def _heartbeat_sender(self):
        """Send periodic heartbeats to maintain connections"""
        while self.running:
            try:
                heartbeat_message = NetworkMessage(
                    sender_id=self.node_id,
                    channel=ChannelType.SYSTEM_COORDINATION,
                    priority=MessagePriority.LOW,
                    payload={
                        'type': 'heartbeat',
                        'timestamp': time.time(),
                        'status': 'active',
                        'load_metrics': self._get_load_metrics()
                    }
                )
                
                await self.send_message(heartbeat_message)
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Heartbeat error: {e}")
                await asyncio.sleep(1)
    
    async def _heartbeat_monitor(self):
        """Monitor heartbeats from other nodes"""
        while self.running:
            try:
                current_time = time.time()
                
                for node_id, node_info in self.nodes.items():
                    if node_id == self.node_id:
                        continue
                    
                    time_since_heartbeat = current_time - node_info.last_heartbeat
                    
                    if time_since_heartbeat > self.failure_threshold:
                        if node_info.status == NodeStatus.ACTIVE:
                            node_info.status = NodeStatus.DEGRADED
                            self.logger.warning(f"⚠️ Node {node_id} degraded")
                        
                        if time_since_heartbeat > self.failure_threshold * 2:
                            node_info.status = NodeStatus.FAILED
                            self.logger.error(f"❌ Node {node_id} failed")
                            await self._handle_node_failure(node_id)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Heartbeat monitor error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_node_failure(self, failed_node_id: str):
        """Handle node failure and initiate recovery"""
        self.logger.info(f"🔄 Handling failure of {failed_node_id}")
        
        # Remove failed connections
        if failed_node_id in self.connections:
            del self.connections[failed_node_id]
        
        # Redistribute workload
        await self._redistribute_workload(failed_node_id)
        
        # Attempt recovery
        asyncio.create_task(self._attempt_node_recovery(failed_node_id))
    
    async def _attempt_node_recovery(self, failed_node_id: str):
        """Attempt to recover failed node"""
        self.logger.info(f"🔄 Attempting recovery of {failed_node_id}")
        
        recovery_start = time.time()
        
        while time.time() - recovery_start < self.recovery_timeout:
            try:
                # Try to re-establish connection
                await self._establish_connection(failed_node_id)
                
                # Send recovery message
                recovery_message = NetworkMessage(
                    sender_id=self.node_id,
                    receiver_id=failed_node_id,
                    channel=ChannelType.SYSTEM_COORDINATION,
                    priority=MessagePriority.URGENT,
                    payload={'type': 'recovery_ping'}
                )
                
                await self.send_message(recovery_message)
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.debug(f"Recovery attempt failed: {e}")
                await asyncio.sleep(2)
        
        self.logger.warning(f"⚠️ Failed to recover {failed_node_id}")
    
    async def _redistribute_workload(self, failed_node_id: str):
        """Redistribute workload from failed node"""
        if failed_node_id not in self.nodes:
            return
        
        failed_node = self.nodes[failed_node_id]
        
        # Find alternative nodes with similar capabilities
        alternative_nodes = []
        for node_id, node_info in self.nodes.items():
            if (node_id != failed_node_id and 
                node_info.status == NodeStatus.ACTIVE and
                any(cap in node_info.capabilities for cap in failed_node.capabilities)):
                alternative_nodes.append(node_id)
        
        if alternative_nodes:
            self.logger.info(f"🔄 Redistributing load to {alternative_nodes}")
            
            # Send workload redistribution message
            redistribution_message = NetworkMessage(
                sender_id=self.node_id,
                channel=ChannelType.SYSTEM_COORDINATION,
                priority=MessagePriority.URGENT,
                payload={
                    'type': 'workload_redistribution',
                    'failed_node': failed_node_id,
                    'alternative_nodes': alternative_nodes,
                    'capabilities': failed_node.capabilities
                }
            )
            
            await self.send_message(redistribution_message)
    
    async def _load_balancer(self):
        """Monitor and balance load across nodes"""
        while self.running:
            try:
                # Collect load metrics from all nodes
                node_loads = {}
                for node_id, node_info in self.nodes.items():
                    if node_info.status == NodeStatus.ACTIVE:
                        node_loads[node_id] = node_info.load_metrics.get('cpu_usage', 0)
                
                # Identify overloaded and underutilized nodes
                if len(node_loads) > 1:
                    avg_load = sum(node_loads.values()) / len(node_loads)
                    overloaded = [nid for nid, load in node_loads.items() if load > avg_load * 1.5]
                    underutilized = [nid for nid, load in node_loads.items() if load < avg_load * 0.5]
                    
                    if overloaded and underutilized:
                        await self._balance_load(overloaded, underutilized)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Load balancer error: {e}")
                await asyncio.sleep(10)
    
    async def _balance_load(self, overloaded: List[str], underutilized: List[str]):
        """Balance load between overloaded and underutilized nodes"""
        self.logger.info(f"⚖️ Balancing load: {overloaded} -> {underutilized}")
        
        balance_message = NetworkMessage(
            sender_id=self.node_id,
            channel=ChannelType.SYSTEM_COORDINATION,
            priority=MessagePriority.STANDARD,
            payload={
                'type': 'load_balance',
                'overloaded_nodes': overloaded,
                'underutilized_nodes': underutilized,
                'timestamp': time.time()
            }
        )
        
        await self.send_message(balance_message)
    
    async def _failure_detector(self):
        """Advanced failure detection using gossip protocol"""
        while self.running:
            try:
                # Gossip node status information
                for peer_id in list(self.connections.keys()):
                    gossip_message = NetworkMessage(
                        sender_id=self.node_id,
                        receiver_id=peer_id,
                        channel=ChannelType.SYSTEM_COORDINATION,
                        priority=MessagePriority.LOW,
                        payload={
                            'type': 'gossip',
                            'known_nodes': {
                                nid: {
                                    'status': ninfo.status.value,
                                    'last_seen': ninfo.last_heartbeat,
                                    'reputation': ninfo.reputation_score
                                }
                                for nid, ninfo in self.nodes.items()
                            }
                        }
                    )
                    
                    await self.send_message(gossip_message)
                
                await asyncio.sleep(10)  # Gossip every 10 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Failure detector error: {e}")
                await asyncio.sleep(5)
    
    async def _consensus_monitor(self):
        """Monitor consensus proposals and clean up expired ones"""
        while self.running:
            try:
                current_time = time.time()
                expired_proposals = []
                
                for proposal_id, proposal in self.consensus_proposals.items():
                    if current_time > proposal.timeout and proposal.status == "pending":
                        proposal.status = "expired"
                        expired_proposals.append(proposal_id)
                
                # Clean up expired proposals
                for proposal_id in expired_proposals:
                    del self.consensus_proposals[proposal_id]
                    self.logger.info(f"⏰ Consensus proposal {proposal_id} expired")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"❌ Consensus monitor error: {e}")
                await asyncio.sleep(5)
    
    def _get_load_metrics(self) -> Dict[str, float]:
        """Get current load metrics for this node"""
        import psutil
        
        return {
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'queue_size': sum(q.qsize() for q in self.connections.values()),
            'message_rate': self.message_stats['received'] / max(1, time.time() - getattr(self, 'start_time', time.time()))
        }
    
    def _sign_message(self, message: NetworkMessage) -> str:
        """Sign message for authentication (simplified)"""
        message_str = f"{message.sender_id}{message.timestamp}{json.dumps(message.payload, sort_keys=True)}"
        return hashlib.sha256(message_str.encode()).hexdigest()[:16]
    
    def _verify_message(self, message: NetworkMessage) -> bool:
        """Verify message signature (simplified)"""
        expected_signature = self._sign_message(message)
        return message.signature == expected_signature
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status"""
        active_nodes = [nid for nid, ninfo in self.nodes.items() if ninfo.status == NodeStatus.ACTIVE]
        
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'network_size': len(self.nodes),
            'active_nodes': len(active_nodes),
            'connections': len(self.connections),
            'message_stats': self.message_stats.copy(),
            'consensus_proposals': len(self.consensus_proposals),
            'status': 'running' if self.running else 'stopped'
        }
    
    async def emergency_broadcast(self, message: str, data: Dict[str, Any] = None):
        """Send emergency broadcast to all nodes"""
        emergency_message = NetworkMessage(
            sender_id=self.node_id,
            channel=ChannelType.EMERGENCY_BROADCAST,
            priority=MessagePriority.EMERGENCY,
            payload={
                'type': 'emergency',
                'message': message,
                'data': data or {},
                'timestamp': time.time()
            }
        )
        
        await self.send_message(emergency_message)
        self.logger.critical(f"🚨 Emergency broadcast: {message}")
    
    async def propose_consensus(self, proposal_type: str, proposal_data: Dict[str, Any], 
                              threshold: float = 0.67) -> str:
        """Propose a consensus decision"""
        proposal_id = str(uuid.uuid4())
        
        proposal_message = NetworkMessage(
            sender_id=self.node_id,
            channel=ChannelType.SYSTEM_COORDINATION,
            priority=MessagePriority.URGENT,
            payload={
                'type': 'proposal',
                'proposal_id': proposal_id,
                'proposal_type': proposal_type,
                'proposal_data': proposal_data,
                'threshold': threshold,
                'timeout': 10  # 10 seconds to vote
            },
            requires_consensus=True
        )
        
        await self.send_message(proposal_message)
        self.logger.info(f"📋 Proposed consensus: {proposal_type} ({proposal_id})")
        
        return proposal_id