#!/usr/bin/env python3
"""
AGI Trading System - Real-time Data Synchronizer
Ensures consistent market data and trading state across all mesh network nodes
using CRDT (Conflict-free Replicated Data Types) and vector clocks.
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict


class SyncEventType(Enum):
    """Types of synchronization events"""
    MARKET_DATA_UPDATE = "market_data_update"
    TRADE_SIGNAL = "trade_signal"
    POSITION_UPDATE = "position_update"
    RISK_METRICS = "risk_metrics"
    SYSTEM_STATE = "system_state"
    NEURAL_PATTERN = "neural_pattern"


@dataclass
class VectorClock:
    """Vector clock for distributed synchronization"""
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: str):
        """Increment clock for node"""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1
    
    def update(self, other_clock: 'VectorClock'):
        """Update with another vector clock"""
        for node_id, timestamp in other_clock.clocks.items():
            self.clocks[node_id] = max(self.clocks.get(node_id, 0), timestamp)
    
    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this event happens before another"""
        return (all(self.clocks.get(node, 0) <= other.clocks.get(node, 0) 
                   for node in set(self.clocks.keys()) | set(other.clocks.keys())) and
                any(self.clocks.get(node, 0) < other.clocks.get(node, 0)
                   for node in set(self.clocks.keys()) | set(other.clocks.keys())))
    
    def concurrent_with(self, other: 'VectorClock') -> bool:
        """Check if events are concurrent"""
        return not (self.happens_before(other) or other.happens_before(self))
    
    def to_dict(self) -> Dict[str, int]:
        return self.clocks.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'VectorClock':
        return cls(clocks=data.copy())


@dataclass
class SyncEvent:
    """Synchronization event with vector clock"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_id: str = ""
    event_type: SyncEventType = SyncEventType.SYSTEM_STATE
    data: Dict[str, Any] = field(default_factory=dict)
    vector_clock: VectorClock = field(default_factory=VectorClock)
    timestamp: float = field(default_factory=time.time)
    ttl: int = 60  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'node_id': self.node_id,
            'event_type': self.event_type.value,
            'data': self.data,
            'vector_clock': self.vector_clock.to_dict(),
            'timestamp': self.timestamp,
            'ttl': self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncEvent':
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            node_id=data.get('node_id', ''),
            event_type=SyncEventType(data.get('event_type', 'system_state')),
            data=data.get('data', {}),
            vector_clock=VectorClock.from_dict(data.get('vector_clock', {})),
            timestamp=data.get('timestamp', time.time()),
            ttl=data.get('ttl', 60)
        )


@dataclass
class MarketDataState:
    """CRDT for market data state"""
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    last_price: float = 0.0
    volume: float = 0.0
    timestamp: float = 0.0
    node_id: str = ""
    vector_clock: VectorClock = field(default_factory=VectorClock)
    
    def merge(self, other: 'MarketDataState') -> 'MarketDataState':
        """Merge with another market data state (CRDT merge)"""
        # Use the most recent data based on timestamp
        if other.timestamp > self.timestamp:
            result = MarketDataState(
                symbol=self.symbol,
                bid=other.bid,
                ask=other.ask,
                last_price=other.last_price,
                volume=other.volume,
                timestamp=other.timestamp,
                node_id=other.node_id,
                vector_clock=VectorClock()
            )
        else:
            result = MarketDataState(
                symbol=self.symbol,
                bid=self.bid,
                ask=self.ask,
                last_price=self.last_price,
                volume=self.volume,
                timestamp=self.timestamp,
                node_id=self.node_id,
                vector_clock=VectorClock()
            )
        
        # Merge vector clocks
        result.vector_clock.update(self.vector_clock)
        result.vector_clock.update(other.vector_clock)
        
        return result


@dataclass
class TradingPositionState:
    """CRDT for trading position state"""
    symbol: str
    position_type: str  # long, short, none
    volume: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    timestamp: float = 0.0
    node_id: str = ""
    vector_clock: VectorClock = field(default_factory=VectorClock)
    
    def merge(self, other: 'TradingPositionState') -> 'TradingPositionState':
        """Merge position states using last-writer-wins"""
        if other.timestamp > self.timestamp:
            result = TradingPositionState(
                symbol=self.symbol,
                position_type=other.position_type,
                volume=other.volume,
                entry_price=other.entry_price,
                current_price=other.current_price,
                unrealized_pnl=other.unrealized_pnl,
                timestamp=other.timestamp,
                node_id=other.node_id,
                vector_clock=VectorClock()
            )
        else:
            result = TradingPositionState(
                symbol=self.symbol,
                position_type=self.position_type,
                volume=self.volume,
                entry_price=self.entry_price,
                current_price=self.current_price,
                unrealized_pnl=self.unrealized_pnl,
                timestamp=self.timestamp,
                node_id=self.node_id,
                vector_clock=VectorClock()
            )
        
        result.vector_clock.update(self.vector_clock)
        result.vector_clock.update(other.vector_clock)
        
        return result


class DataSynchronizer:
    """
    Real-time data synchronizer for mesh network
    Implements CRDT-based consistency and vector clock ordering
    """
    
    def __init__(self, node_id: str, mesh_coordinator):
        self.node_id = node_id
        self.mesh_coordinator = mesh_coordinator
        self.logger = logging.getLogger(f"DataSync.{node_id}")
        
        # State management
        self.vector_clock = VectorClock()
        self.event_log: List[SyncEvent] = []
        self.market_data: Dict[str, MarketDataState] = {}
        self.position_states: Dict[str, TradingPositionState] = {}
        self.neural_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Synchronization tracking
        self.pending_events: Dict[str, SyncEvent] = {}
        self.sync_stats = {
            'events_processed': 0,
            'conflicts_resolved': 0,
            'sync_operations': 0,
            'data_integrity_checks': 0
        }
        
        # Anti-entropy gossip
        self.gossip_interval = 5.0  # seconds
        self.max_event_age = 300  # 5 minutes
        
        # Running state
        self.running = False
        self.tasks = []
        
        self.logger.info(f"📊 Data synchronizer initialized for {node_id}")
    
    async def start(self):
        """Start the data synchronizer"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("🚀 Starting data synchronizer...")
        
        # Register message handlers
        from .mesh_network_coordinator import ChannelType
        self.mesh_coordinator.register_message_handler(
            ChannelType.MARKET_DATA, self._handle_market_data_message
        )
        self.mesh_coordinator.register_message_handler(
            ChannelType.NEURAL_PATTERNS, self._handle_neural_pattern_message
        )
        self.mesh_coordinator.register_message_handler(
            ChannelType.SYSTEM_COORDINATION, self._handle_sync_message
        )
        
        # Start synchronization tasks
        self.tasks = [
            asyncio.create_task(self._anti_entropy_gossip()),
            asyncio.create_task(self._event_log_cleaner()),
            asyncio.create_task(self._data_integrity_checker()),
            asyncio.create_task(self._conflict_resolver())
        ]
        
        self.logger.info("✅ Data synchronizer started")
    
    async def stop(self):
        """Stop the data synchronizer"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("🛑 Stopping data synchronizer...")
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.logger.info("✅ Data synchronizer stopped")
    
    async def sync_market_data(self, symbol: str, bid: float, ask: float, 
                              last_price: float, volume: float):
        """Synchronize market data across network"""
        # Update vector clock
        self.vector_clock.increment(self.node_id)
        
        # Create market data state
        market_state = MarketDataState(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last_price=last_price,
            volume=volume,
            timestamp=time.time(),
            node_id=self.node_id,
            vector_clock=VectorClock(self.vector_clock.clocks.copy())
        )
        
        # Update local state
        if symbol in self.market_data:
            self.market_data[symbol] = self.market_data[symbol].merge(market_state)
        else:
            self.market_data[symbol] = market_state
        
        # Create sync event
        sync_event = SyncEvent(
            node_id=self.node_id,
            event_type=SyncEventType.MARKET_DATA_UPDATE,
            data={
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'last_price': last_price,
                'volume': volume,
                'market_timestamp': time.time()
            },
            vector_clock=VectorClock(self.vector_clock.clocks.copy())
        )
        
        # Add to event log
        self.event_log.append(sync_event)
        
        # Broadcast to network
        await self._broadcast_sync_event(sync_event)
        
        self.logger.debug(f"📊 Synced market data for {symbol}")
    
    async def sync_position_update(self, symbol: str, position_type: str, 
                                  volume: float, entry_price: float, 
                                  current_price: float, unrealized_pnl: float):
        """Synchronize position updates across network"""
        # Update vector clock
        self.vector_clock.increment(self.node_id)
        
        # Create position state
        position_state = TradingPositionState(
            symbol=symbol,
            position_type=position_type,
            volume=volume,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            timestamp=time.time(),
            node_id=self.node_id,
            vector_clock=VectorClock(self.vector_clock.clocks.copy())
        )
        
        # Update local state
        if symbol in self.position_states:
            self.position_states[symbol] = self.position_states[symbol].merge(position_state)
        else:
            self.position_states[symbol] = position_state
        
        # Create sync event
        sync_event = SyncEvent(
            node_id=self.node_id,
            event_type=SyncEventType.POSITION_UPDATE,
            data={
                'symbol': symbol,
                'position_type': position_type,
                'volume': volume,
                'entry_price': entry_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'position_timestamp': time.time()
            },
            vector_clock=VectorClock(self.vector_clock.clocks.copy())
        )
        
        # Add to event log
        self.event_log.append(sync_event)
        
        # Broadcast to network
        await self._broadcast_sync_event(sync_event)
        
        self.logger.debug(f"💼 Synced position update for {symbol}")
    
    async def sync_neural_pattern(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """Synchronize neural patterns for distributed learning"""
        # Update vector clock
        self.vector_clock.increment(self.node_id)
        
        # Store neural pattern
        self.neural_patterns[pattern_id] = {
            'data': pattern_data,
            'timestamp': time.time(),
            'node_id': self.node_id,
            'vector_clock': self.vector_clock.clocks.copy()
        }
        
        # Create sync event
        sync_event = SyncEvent(
            node_id=self.node_id,
            event_type=SyncEventType.NEURAL_PATTERN,
            data={
                'pattern_id': pattern_id,
                'pattern_data': pattern_data,
                'learning_timestamp': time.time()
            },
            vector_clock=VectorClock(self.vector_clock.clocks.copy())
        )
        
        # Add to event log
        self.event_log.append(sync_event)
        
        # Broadcast to neural agents
        await self._broadcast_neural_pattern(sync_event)
        
        self.logger.debug(f"🧠 Synced neural pattern {pattern_id}")
    
    async def _broadcast_sync_event(self, event: SyncEvent):
        """Broadcast synchronization event to network"""
        from .mesh_network_coordinator import NetworkMessage, ChannelType, MessagePriority
        
        message = NetworkMessage(
            sender_id=self.node_id,
            channel=ChannelType.MARKET_DATA,
            priority=MessagePriority.STANDARD,
            payload={
                'type': 'sync_event',
                'event': event.to_dict()
            }
        )
        
        await self.mesh_coordinator.send_message(message)
        self.sync_stats['sync_operations'] += 1
    
    async def _broadcast_neural_pattern(self, event: SyncEvent):
        """Broadcast neural pattern to ML agents"""
        from .mesh_network_coordinator import NetworkMessage, ChannelType, MessagePriority
        
        message = NetworkMessage(
            sender_id=self.node_id,
            channel=ChannelType.NEURAL_PATTERNS,
            priority=MessagePriority.STANDARD,
            payload={
                'type': 'neural_pattern',
                'event': event.to_dict()
            }
        )
        
        await self.mesh_coordinator.send_message(message)
    
    async def _handle_market_data_message(self, message):
        """Handle incoming market data sync messages"""
        try:
            payload = message.payload
            
            if payload.get('type') == 'sync_event':
                event = SyncEvent.from_dict(payload['event'])
                await self._process_sync_event(event)
            
        except Exception as e:
            self.logger.error(f"❌ Error handling market data message: {e}")
    
    async def _handle_neural_pattern_message(self, message):
        """Handle incoming neural pattern messages"""
        try:
            payload = message.payload
            
            if payload.get('type') == 'neural_pattern':
                event = SyncEvent.from_dict(payload['event'])
                await self._process_neural_pattern_event(event)
            
        except Exception as e:
            self.logger.error(f"❌ Error handling neural pattern message: {e}")
    
    async def _handle_sync_message(self, message):
        """Handle synchronization coordination messages"""
        try:
            payload = message.payload
            
            if payload.get('type') == 'gossip_request':
                await self._handle_gossip_request(message)
            elif payload.get('type') == 'gossip_response':
                await self._handle_gossip_response(message)
            
        except Exception as e:
            self.logger.error(f"❌ Error handling sync message: {e}")
    
    async def _process_sync_event(self, event: SyncEvent):
        """Process incoming synchronization event"""
        # Check if event is valid and not expired
        if time.time() - event.timestamp > event.ttl:
            return
        
        # Update vector clock
        self.vector_clock.update(event.vector_clock)
        
        # Process based on event type
        if event.event_type == SyncEventType.MARKET_DATA_UPDATE:
            await self._process_market_data_event(event)
        elif event.event_type == SyncEventType.POSITION_UPDATE:
            await self._process_position_event(event)
        elif event.event_type == SyncEventType.RISK_METRICS:
            await self._process_risk_metrics_event(event)
        
        # Add to event log if not already present
        if not any(e.id == event.id for e in self.event_log):
            self.event_log.append(event)
        
        self.sync_stats['events_processed'] += 1
    
    async def _process_market_data_event(self, event: SyncEvent):
        """Process market data synchronization event"""
        data = event.data
        symbol = data['symbol']
        
        # Create market data state from event
        new_state = MarketDataState(
            symbol=symbol,
            bid=data['bid'],
            ask=data['ask'],
            last_price=data['last_price'],
            volume=data['volume'],
            timestamp=data['market_timestamp'],
            node_id=event.node_id,
            vector_clock=event.vector_clock
        )
        
        # Merge with existing state
        if symbol in self.market_data:
            old_state = self.market_data[symbol]
            merged_state = old_state.merge(new_state)
            
            # Check for conflicts
            if (old_state.timestamp != merged_state.timestamp or 
                old_state.node_id != merged_state.node_id):
                self.sync_stats['conflicts_resolved'] += 1
                self.logger.debug(f"🔧 Resolved conflict for {symbol}")
            
            self.market_data[symbol] = merged_state
        else:
            self.market_data[symbol] = new_state
    
    async def _process_position_event(self, event: SyncEvent):
        """Process position update synchronization event"""
        data = event.data
        symbol = data['symbol']
        
        # Create position state from event
        new_state = TradingPositionState(
            symbol=symbol,
            position_type=data['position_type'],
            volume=data['volume'],
            entry_price=data['entry_price'],
            current_price=data['current_price'],
            unrealized_pnl=data['unrealized_pnl'],
            timestamp=data['position_timestamp'],
            node_id=event.node_id,
            vector_clock=event.vector_clock
        )
        
        # Merge with existing state
        if symbol in self.position_states:
            old_state = self.position_states[symbol]
            merged_state = old_state.merge(new_state)
            
            if (old_state.timestamp != merged_state.timestamp or 
                old_state.node_id != merged_state.node_id):
                self.sync_stats['conflicts_resolved'] += 1
            
            self.position_states[symbol] = merged_state
        else:
            self.position_states[symbol] = new_state
    
    async def _process_neural_pattern_event(self, event: SyncEvent):
        """Process neural pattern synchronization event"""
        data = event.data
        pattern_id = data['pattern_id']
        
        # Store or update neural pattern
        existing_pattern = self.neural_patterns.get(pattern_id)
        
        if (not existing_pattern or 
            data['learning_timestamp'] > existing_pattern.get('timestamp', 0)):
            
            self.neural_patterns[pattern_id] = {
                'data': data['pattern_data'],
                'timestamp': data['learning_timestamp'],
                'node_id': event.node_id,
                'vector_clock': event.vector_clock.clocks.copy()
            }
            
            self.logger.debug(f"🧠 Updated neural pattern {pattern_id}")
    
    async def _process_risk_metrics_event(self, event: SyncEvent):
        """Process risk metrics synchronization event"""
        # Implementation for risk metrics synchronization
        pass
    
    async def _anti_entropy_gossip(self):
        """Anti-entropy gossip protocol for eventual consistency"""
        while self.running:
            try:
                # Select random peers for gossip
                peers = list(self.mesh_coordinator.connections.keys())
                if peers:
                    import random
                    peer = random.choice(peers)
                    await self._initiate_gossip(peer)
                
                await asyncio.sleep(self.gossip_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Gossip error: {e}")
                await asyncio.sleep(1)
    
    async def _initiate_gossip(self, peer_id: str):
        """Initiate gossip with a peer"""
        from .mesh_network_coordinator import NetworkMessage, ChannelType, MessagePriority
        
        # Send gossip request with our event summary
        event_summary = {
            'vector_clock': self.vector_clock.to_dict(),
            'event_count': len(self.event_log),
            'latest_events': [
                {'id': e.id, 'timestamp': e.timestamp, 'type': e.event_type.value}
                for e in sorted(self.event_log, key=lambda x: x.timestamp)[-10:]
            ]
        }
        
        gossip_message = NetworkMessage(
            sender_id=self.node_id,
            receiver_id=peer_id,
            channel=ChannelType.SYSTEM_COORDINATION,
            priority=MessagePriority.LOW,
            payload={
                'type': 'gossip_request',
                'event_summary': event_summary
            }
        )
        
        await self.mesh_coordinator.send_message(gossip_message)
    
    async def _handle_gossip_request(self, message):
        """Handle incoming gossip request"""
        from .mesh_network_coordinator import NetworkMessage, ChannelType, MessagePriority
        
        peer_summary = message.payload['event_summary']
        peer_vector_clock = VectorClock.from_dict(peer_summary['vector_clock'])
        
        # Find events that peer might be missing
        missing_events = []
        for event in self.event_log:
            if event.vector_clock.happens_before(peer_vector_clock):
                continue  # Peer already has this
            missing_events.append(event.to_dict())
        
        # Send response with missing events
        response_message = NetworkMessage(
            sender_id=self.node_id,
            receiver_id=message.sender_id,
            channel=ChannelType.SYSTEM_COORDINATION,
            priority=MessagePriority.LOW,
            payload={
                'type': 'gossip_response',
                'missing_events': missing_events[:50],  # Limit to 50 events
                'vector_clock': self.vector_clock.to_dict()
            }
        )
        
        await self.mesh_coordinator.send_message(response_message)
    
    async def _handle_gossip_response(self, message):
        """Handle gossip response with missing events"""
        missing_events = message.payload['missing_events']
        peer_vector_clock = VectorClock.from_dict(message.payload['vector_clock'])
        
        # Process missing events
        for event_data in missing_events:
            event = SyncEvent.from_dict(event_data)
            await self._process_sync_event(event)
        
        # Update our vector clock
        self.vector_clock.update(peer_vector_clock)
    
    async def _event_log_cleaner(self):
        """Clean up old events from the event log"""
        while self.running:
            try:
                current_time = time.time()
                
                # Remove events older than max_event_age
                self.event_log = [
                    event for event in self.event_log
                    if current_time - event.timestamp < self.max_event_age
                ]
                
                await asyncio.sleep(60)  # Clean every minute
                
            except Exception as e:
                self.logger.error(f"❌ Event log cleaner error: {e}")
                await asyncio.sleep(10)
    
    async def _data_integrity_checker(self):
        """Check data integrity across synchronized states"""
        while self.running:
            try:
                # Check market data consistency
                for symbol, market_state in self.market_data.items():
                    if time.time() - market_state.timestamp > 60:  # Data too old
                        self.logger.warning(f"⚠️ Stale market data for {symbol}")
                
                # Check position consistency
                for symbol, position_state in self.position_states.items():
                    if position_state.volume < 0:
                        self.logger.warning(f"⚠️ Invalid position volume for {symbol}")
                
                self.sync_stats['data_integrity_checks'] += 1
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Data integrity checker error: {e}")
                await asyncio.sleep(10)
    
    async def _conflict_resolver(self):
        """Resolve data conflicts using CRDT merge operations"""
        while self.running:
            try:
                # Check for pending events that might need conflict resolution
                current_time = time.time()
                
                for event in list(self.pending_events.values()):
                    if current_time - event.timestamp > 10:  # Process delayed events
                        await self._process_sync_event(event)
                        del self.pending_events[event.id]
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"❌ Conflict resolver error: {e}")
                await asyncio.sleep(5)
    
    def get_market_data(self, symbol: str) -> Optional[MarketDataState]:
        """Get current market data for symbol"""
        return self.market_data.get(symbol)
    
    def get_position_state(self, symbol: str) -> Optional[TradingPositionState]:
        """Get current position state for symbol"""
        return self.position_states.get(symbol)
    
    def get_neural_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get neural pattern by ID"""
        return self.neural_patterns.get(pattern_id)
    
    def get_all_neural_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get all neural patterns"""
        return self.neural_patterns.copy()
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status"""
        return {
            'node_id': self.node_id,
            'vector_clock': self.vector_clock.to_dict(),
            'event_log_size': len(self.event_log),
            'market_data_symbols': len(self.market_data),
            'position_symbols': len(self.position_states),
            'neural_patterns': len(self.neural_patterns),
            'sync_stats': self.sync_stats.copy(),
            'running': self.running
        }
    
    async def force_sync(self):
        """Force synchronization with all peers"""
        self.logger.info("🔄 Forcing synchronization with all peers...")
        
        peers = list(self.mesh_coordinator.connections.keys())
        for peer in peers:
            await self._initiate_gossip(peer)
        
        self.logger.info(f"✅ Initiated sync with {len(peers)} peers")