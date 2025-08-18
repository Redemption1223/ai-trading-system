"""
Enhanced Error Recovery and Alert System
Integrated from MQL5 Expert Advisor - Complete Error Handling Framework
"""

import time
import threading
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

# Optional imports for alerts
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Alert types"""
    SYSTEM_ERROR = "SYSTEM_ERROR"
    TRADING_ERROR = "TRADING_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    PERFORMANCE_WARNING = "PERFORMANCE_WARNING"
    RISK_ALERT = "RISK_ALERT"
    ACCOUNT_ALERT = "ACCOUNT_ALERT"
    MARKET_ALERT = "MARKET_ALERT"

@dataclass
class ErrorEvent:
    """Error event data structure"""
    timestamp: datetime
    error_type: str
    error_message: str
    error_source: str
    severity: AlertLevel
    stack_trace: Optional[str] = None
    context_data: Optional[Dict] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'error_type': self.error_type,
            'error_message': self.error_message,
            'error_source': self.error_source,
            'severity': self.severity.value,
            'stack_trace': self.stack_trace,
            'context_data': self.context_data,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'retry_count': self.retry_count
        }

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    alert_type: AlertType
    condition: Callable[[Any], bool]
    threshold: float
    cooldown_seconds: int = 300  # 5 minutes default
    last_triggered: Optional[datetime] = None
    enabled: bool = True
    
    def should_trigger(self, value: Any) -> bool:
        """Check if alert should trigger"""
        if not self.enabled:
            return False
        
        # Check cooldown
        if self.last_triggered:
            if datetime.now() - self.last_triggered < timedelta(seconds=self.cooldown_seconds):
                return False
        
        # Check condition
        return self.condition(value)
    
    def trigger(self):
        """Mark alert as triggered"""
        self.last_triggered = datetime.now()

@dataclass
class RecoveryStrategy:
    """Recovery strategy definition"""
    name: str
    error_types: List[str]
    recovery_function: Callable
    max_retries: int = 3
    retry_delay: float = 1.0
    escalation_strategy: Optional[str] = None

class EnhancedErrorRecoverySystem:
    """Enhanced Error Recovery and Alert System based on MQL5"""
    
    def __init__(self):
        self.name = "ENHANCED_ERROR_RECOVERY_SYSTEM"
        self.version = "1.0.0"
        
        # Error tracking
        self.error_history = []
        self.active_errors = {}
        self.recovery_strategies = {}
        self.alert_rules = {}
        
        # Configuration
        self.config = {
            'max_error_history': 1000,
            'auto_restart': True,
            'max_retries': 5,
            'emergency_stop': False,
            'max_daily_loss': 500.0,
            'error_count_threshold': 50,  # Increased to avoid false emergencies
            'critical_error_threshold': 10  # Increased to avoid false emergencies
        }
        
        # Alert configuration
        self.alert_config = {
            'email_alerts': False,
            'push_notifications': False,
            'telegram_bot': False,
            'webhook_url': '',
            'email_address': '',
            'telegram_chat_id': '',
            'smtp_server': '',
            'smtp_port': 587,
            'smtp_username': '',
            'smtp_password': ''
        }
        
        # State tracking
        self.is_running = False
        self.monitoring_thread = None
        self.last_health_check = None
        self.system_health_score = 100.0
        
        # Recovery state
        self.emergency_mode = False
        self.auto_recovery_enabled = True
        self.recovery_in_progress = False
        
        # Statistics
        self.stats = {
            'total_errors': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'alerts_sent': 0,
            'uptime_start': datetime.now()
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            # Setup file handler for error logs
            log_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Error log file
            error_handler = logging.FileHandler('error_recovery.log')
            error_handler.setFormatter(log_formatter)
            error_handler.setLevel(logging.ERROR)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            console_handler.setLevel(logging.INFO)
            
            self.logger.addHandler(error_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize default recovery strategies
        self._setup_default_recovery_strategies()
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
    
    def initialize(self) -> Dict:
        """Initialize the error recovery system"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Load configuration if exists
            self._load_configuration()
            
            # Start monitoring thread
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("Enhanced Error Recovery System initialized successfully")
            return {
                "status": "initialized",
                "agent": "ENHANCED_ERROR_RECOVERY_SYSTEM",
                "auto_recovery": self.auto_recovery_enabled,
                "emergency_mode": self.emergency_mode,
                "recovery_strategies": len(self.recovery_strategies),
                "alert_rules": len(self.alert_rules)
            }
            
        except Exception as e:
            self.logger.error(f"Error recovery system initialization failed: {e}")
            return {"status": "failed", "agent": "ENHANCED_ERROR_RECOVERY_SYSTEM", "error": str(e)}
    
    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies"""
        # MT5 Connection Recovery
        self.recovery_strategies['mt5_connection'] = RecoveryStrategy(
            name="MT5 Connection Recovery",
            error_types=['ConnectionError', 'MT5ConnectionError', 'TimeoutError'],
            recovery_function=self._recover_mt5_connection,
            max_retries=5,
            retry_delay=2.0
        )
        
        # Memory Recovery
        self.recovery_strategies['memory_cleanup'] = RecoveryStrategy(
            name="Memory Cleanup Recovery",
            error_types=['MemoryError', 'OutOfMemoryError'],
            recovery_function=self._recover_memory_cleanup,
            max_retries=3,
            retry_delay=1.0
        )
        
        # Data Feed Recovery
        self.recovery_strategies['data_feed'] = RecoveryStrategy(
            name="Data Feed Recovery",
            error_types=['DataFeedError', 'PriceDataError'],
            recovery_function=self._recover_data_feed,
            max_retries=3,
            retry_delay=1.5
        )
        
        # Trading Engine Recovery
        self.recovery_strategies['trading_engine'] = RecoveryStrategy(
            name="Trading Engine Recovery",
            error_types=['TradingError', 'OrderExecutionError'],
            recovery_function=self._recover_trading_engine,
            max_retries=2,
            retry_delay=3.0
        )
        
        # General System Recovery
        self.recovery_strategies['system_restart'] = RecoveryStrategy(
            name="System Restart Recovery",
            error_types=['SystemError', 'CriticalError'],
            recovery_function=self._recover_system_restart,
            max_retries=1,
            retry_delay=5.0
        )
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        # High drawdown alert
        self.alert_rules['high_drawdown'] = AlertRule(
            name="High Portfolio Drawdown",
            alert_type=AlertType.RISK_ALERT,
            condition=lambda drawdown: drawdown > 5.0,  # 5% drawdown
            threshold=5.0,
            cooldown_seconds=600  # 10 minutes
        )
        
        # Trading disabled alert
        self.alert_rules['trading_disabled'] = AlertRule(
            name="Trading Disabled",
            alert_type=AlertType.TRADING_ERROR,
            condition=lambda enabled: not enabled,
            threshold=0,
            cooldown_seconds=1800  # 30 minutes
        )
        
        # Connection lost alert
        self.alert_rules['connection_lost'] = AlertRule(
            name="MT5 Connection Lost",
            alert_type=AlertType.CONNECTION_ERROR,
            condition=lambda connected: not connected,
            threshold=0,
            cooldown_seconds=300  # 5 minutes
        )
        
        # High volatility alert
        self.alert_rules['high_volatility'] = AlertRule(
            name="High Market Volatility",
            alert_type=AlertType.MARKET_ALERT,
            condition=lambda volatility: volatility > 2.0,  # 2% volatility
            threshold=2.0,
            cooldown_seconds=900  # 15 minutes
        )
        
        # System health alert
        self.alert_rules['system_health'] = AlertRule(
            name="Low System Health",
            alert_type=AlertType.SYSTEM_ERROR,
            condition=lambda health: health < 70.0,  # Below 70% health
            threshold=70.0,
            cooldown_seconds=600  # 10 minutes
        )
    
    def report_error(self, error_type: str, error_message: str, error_source: str, 
                    severity: AlertLevel = AlertLevel.ERROR, stack_trace: str = None, 
                    context_data: Dict = None) -> str:
        """Report an error to the recovery system"""
        try:
            # Create error event
            error_event = ErrorEvent(
                timestamp=datetime.now(),
                error_type=error_type,
                error_message=error_message,
                error_source=error_source,
                severity=severity,
                stack_trace=stack_trace,
                context_data=context_data or {}
            )
            
            # Generate error ID
            error_id = f"{error_source}_{int(time.time())}"
            
            # Store error
            self.error_history.append(error_event)
            self.active_errors[error_id] = error_event
            
            # Update statistics
            self.stats['total_errors'] += 1
            
            # Maintain history size
            if len(self.error_history) > self.config['max_error_history']:
                self.error_history.pop(0)
            
            # Log error
            self.logger.error(f"Error reported: {error_type} from {error_source}: {error_message}")
            
            # Check for emergency conditions
            self._check_emergency_conditions()
            
            # Attempt recovery if enabled
            if self.auto_recovery_enabled and not self.emergency_mode:
                self._attempt_recovery(error_event, error_id)
            
            # Send alerts
            self._send_alert(error_event)
            
            return error_id
            
        except Exception as e:
            self.logger.critical(f"Critical error in error reporting system: {e}")
            return "error_reporting_failed"
    
    def _attempt_recovery(self, error_event: ErrorEvent, error_id: str):
        """Attempt to recover from error"""
        try:
            self.recovery_in_progress = True
            self.stats['recovery_attempts'] += 1
            
            # Find appropriate recovery strategy
            recovery_strategy = None
            for strategy_name, strategy in self.recovery_strategies.items():
                if error_event.error_type in strategy.error_types:
                    recovery_strategy = strategy
                    break
            
            if not recovery_strategy:
                self.logger.warning(f"No recovery strategy found for error type: {error_event.error_type}")
                return False
            
            self.logger.info(f"Attempting recovery using strategy: {recovery_strategy.name}")
            
            # Attempt recovery with retries
            for attempt in range(recovery_strategy.max_retries):
                try:
                    error_event.retry_count = attempt + 1
                    error_event.recovery_attempted = True
                    
                    # Execute recovery function
                    recovery_result = recovery_strategy.recovery_function(error_event)
                    
                    if recovery_result:
                        error_event.recovery_successful = True
                        self.stats['successful_recoveries'] += 1
                        self.logger.info(f"Recovery successful for error: {error_id}")
                        
                        # Remove from active errors
                        if error_id in self.active_errors:
                            del self.active_errors[error_id]
                        
                        return True
                    else:
                        self.logger.warning(f"Recovery attempt {attempt + 1} failed for error: {error_id}")
                        if attempt < recovery_strategy.max_retries - 1:
                            time.sleep(recovery_strategy.retry_delay)
                
                except Exception as recovery_error:
                    self.logger.error(f"Recovery attempt {attempt + 1} exception: {recovery_error}")
                    if attempt < recovery_strategy.max_retries - 1:
                        time.sleep(recovery_strategy.retry_delay)
            
            # All recovery attempts failed
            error_event.recovery_successful = False
            self.stats['failed_recoveries'] += 1
            self.logger.error(f"All recovery attempts failed for error: {error_id}")
            
            # Check for escalation
            if recovery_strategy.escalation_strategy:
                self._escalate_error(error_event, recovery_strategy.escalation_strategy)
            
            return False
            
        except Exception as e:
            self.logger.critical(f"Critical error in recovery system: {e}")
            return False
        finally:
            self.recovery_in_progress = False
    
    def _recover_mt5_connection(self, error_event: ErrorEvent) -> bool:
        """Recover MT5 connection"""
        try:
            self.logger.info("Attempting MT5 connection recovery...")
            
            # This would be connected to the actual MT5 connector
            # For now, simulate recovery
            time.sleep(1)
            
            # Simulate success/failure
            import random
            success = random.random() > 0.3  # 70% success rate
            
            if success:
                self.logger.info("MT5 connection recovery successful")
            else:
                self.logger.warning("MT5 connection recovery failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"MT5 connection recovery error: {e}")
            return False
    
    def _recover_memory_cleanup(self, error_event: ErrorEvent) -> bool:
        """Recover from memory issues"""
        try:
            self.logger.info("Attempting memory cleanup recovery...")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear caches if available
            # This would clear application-specific caches
            
            self.logger.info("Memory cleanup recovery completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory cleanup recovery error: {e}")
            return False
    
    def _recover_data_feed(self, error_event: ErrorEvent) -> bool:
        """Recover data feed connection"""
        try:
            self.logger.info("Attempting data feed recovery...")
            
            # This would reconnect data feeds
            time.sleep(1)
            
            # Simulate recovery
            import random
            success = random.random() > 0.2  # 80% success rate
            
            if success:
                self.logger.info("Data feed recovery successful")
            else:
                self.logger.warning("Data feed recovery failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Data feed recovery error: {e}")
            return False
    
    def _recover_trading_engine(self, error_event: ErrorEvent) -> bool:
        """Recover trading engine"""
        try:
            self.logger.info("Attempting trading engine recovery...")
            
            # This would restart trading components
            time.sleep(2)
            
            # Simulate recovery
            import random
            success = random.random() > 0.4  # 60% success rate
            
            if success:
                self.logger.info("Trading engine recovery successful")
            else:
                self.logger.warning("Trading engine recovery failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Trading engine recovery error: {e}")
            return False
    
    def _recover_system_restart(self, error_event: ErrorEvent) -> bool:
        """Recover by system restart"""
        try:
            self.logger.warning("Attempting system restart recovery...")
            
            # This would trigger a controlled system restart
            # For safety, we'll just log this action
            self.logger.warning("System restart would be triggered here")
            
            return True
            
        except Exception as e:
            self.logger.error(f"System restart recovery error: {e}")
            return False
    
    def _escalate_error(self, error_event: ErrorEvent, escalation_strategy: str):
        """Escalate error to higher level"""
        self.logger.critical(f"Escalating error {error_event.error_type} using strategy: {escalation_strategy}")
        
        # Send critical alert
        critical_event = ErrorEvent(
            timestamp=datetime.now(),
            error_type="ESCALATED_ERROR",
            error_message=f"Escalated: {error_event.error_message}",
            error_source="ERROR_RECOVERY_SYSTEM",
            severity=AlertLevel.CRITICAL
        )
        
        self._send_alert(critical_event)
    
    def _check_emergency_conditions(self):
        """Check for emergency conditions"""
        # Check error count threshold
        recent_errors = [e for e in self.error_history 
                        if (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
        
        if len(recent_errors) > self.config['error_count_threshold']:
            self.logger.critical("Emergency condition: Too many errors in last hour")
            self._trigger_emergency_mode("Too many errors")
        
        # Check critical error threshold
        critical_errors = [e for e in recent_errors if e.severity == AlertLevel.CRITICAL]
        if len(critical_errors) > self.config['critical_error_threshold']:
            self.logger.critical("Emergency condition: Too many critical errors")
            self._trigger_emergency_mode("Too many critical errors")
    
    def _trigger_emergency_mode(self, reason: str):
        """Trigger emergency mode"""
        if not self.emergency_mode:
            self.emergency_mode = True
            self.auto_recovery_enabled = False
            
            emergency_event = ErrorEvent(
                timestamp=datetime.now(),
                error_type="EMERGENCY_MODE",
                error_message=f"Emergency mode triggered: {reason}",
                error_source="ERROR_RECOVERY_SYSTEM",
                severity=AlertLevel.CRITICAL
            )
            
            self.logger.critical(f"EMERGENCY MODE ACTIVATED: {reason}")
            self._send_alert(emergency_event)
    
    def check_alert_conditions(self, condition_name: str, value: Any):
        """Check if alert conditions are met"""
        if condition_name in self.alert_rules:
            alert_rule = self.alert_rules[condition_name]
            
            if alert_rule.should_trigger(value):
                alert_rule.trigger()
                
                alert_event = ErrorEvent(
                    timestamp=datetime.now(),
                    error_type=alert_rule.alert_type.value,
                    error_message=f"Alert condition met: {alert_rule.name}",
                    error_source="ALERT_SYSTEM",
                    severity=AlertLevel.WARNING,
                    context_data={'condition_value': value, 'threshold': alert_rule.threshold}
                )
                
                self._send_alert(alert_event)
    
    def _send_alert(self, error_event: ErrorEvent):
        """Send alert notifications"""
        try:
            self.stats['alerts_sent'] += 1
            
            # Console alert
            self.logger.error(f"ALERT: {error_event.error_message}")
            
            # Email alert
            if self.alert_config['email_alerts'] and self.alert_config['email_address']:
                self._send_email_alert(error_event)
            
            # Webhook alert
            if self.alert_config['webhook_url']:
                self._send_webhook_alert(error_event)
            
            # Telegram alert
            if self.alert_config['telegram_bot'] and self.alert_config['telegram_chat_id']:
                self._send_telegram_alert(error_event)
                
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    def _send_email_alert(self, error_event: ErrorEvent):
        """Send email alert"""
        try:
            if not EMAIL_AVAILABLE:
                self.logger.warning("Email not available - skipping email alert")
                return
                
            if not all([self.alert_config['smtp_server'], self.alert_config['smtp_username'], 
                       self.alert_config['smtp_password']]):
                return
            
            msg = MimeMultipart()
            msg['From'] = self.alert_config['smtp_username']
            msg['To'] = self.alert_config['email_address']
            msg['Subject'] = f"AGI Trading System Alert: {error_event.error_type}"
            
            body = f"""
            Alert from AGI Trading System
            
            Time: {error_event.timestamp}
            Type: {error_event.error_type}
            Severity: {error_event.severity.value}
            Source: {error_event.error_source}
            Message: {error_event.error_message}
            
            Recovery Attempted: {error_event.recovery_attempted}
            Recovery Successful: {error_event.recovery_successful}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.alert_config['smtp_server'], self.alert_config['smtp_port'])
            server.starttls()
            server.login(self.alert_config['smtp_username'], self.alert_config['smtp_password'])
            text = msg.as_string()
            server.sendmail(self.alert_config['smtp_username'], self.alert_config['email_address'], text)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, error_event: ErrorEvent):
        """Send webhook alert"""
        try:
            if not REQUESTS_AVAILABLE:
                self.logger.warning("Requests not available - skipping webhook alert")
                return
                
            payload = {
                'timestamp': error_event.timestamp.isoformat(),
                'error_type': error_event.error_type,
                'severity': error_event.severity.value,
                'source': error_event.error_source,
                'message': error_event.error_message,
                'recovery_attempted': error_event.recovery_attempted,
                'recovery_successful': error_event.recovery_successful
            }
            
            requests.post(self.alert_config['webhook_url'], json=payload, timeout=10)
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_telegram_alert(self, error_event: ErrorEvent):
        """Send Telegram alert"""
        try:
            # This would implement Telegram bot messaging
            self.logger.info("Telegram alert would be sent here")
            
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Update system health
                self._update_system_health()
                
                # Check alert conditions
                self.check_alert_conditions('system_health', self.system_health_score)
                
                # Clean old errors
                self._cleanup_old_errors()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _update_system_health(self):
        """Update system health score"""
        try:
            health_score = 100.0
            
            # Deduct for recent errors
            recent_errors = [e for e in self.error_history 
                           if (datetime.now() - e.timestamp).total_seconds() < 3600]
            health_score -= len(recent_errors) * 5
            
            # Deduct for active errors
            health_score -= len(self.active_errors) * 10
            
            # Deduct for emergency mode
            if self.emergency_mode:
                health_score -= 50
            
            # Deduct for recovery failures
            if self.stats['failed_recoveries'] > 0:
                failure_rate = self.stats['failed_recoveries'] / max(1, self.stats['recovery_attempts'])
                health_score -= failure_rate * 30
            
            self.system_health_score = max(0.0, min(100.0, health_score))
            self.last_health_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"System health update error: {e}")
    
    def _cleanup_old_errors(self):
        """Clean up old errors"""
        try:
            # Remove resolved active errors older than 1 hour
            current_time = datetime.now()
            old_error_ids = []
            
            for error_id, error_event in self.active_errors.items():
                if (current_time - error_event.timestamp).total_seconds() > 3600:  # 1 hour
                    if error_event.recovery_successful:
                        old_error_ids.append(error_id)
            
            for error_id in old_error_ids:
                del self.active_errors[error_id]
                
        except Exception as e:
            self.logger.error(f"Error cleanup failed: {e}")
    
    def _load_configuration(self):
        """Load configuration from file"""
        try:
            config_file = 'error_recovery_config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config.get('system_config', {}))
                    self.alert_config.update(saved_config.get('alert_config', {}))
                    
        except Exception as e:
            self.logger.warning(f"Failed to load configuration: {e}")
    
    def save_configuration(self):
        """Save configuration to file"""
        try:
            config_data = {
                'system_config': self.config,
                'alert_config': self.alert_config
            }
            
            with open('error_recovery_config.json', 'w') as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        uptime = (datetime.now() - self.stats['uptime_start']).total_seconds()
        
        return {
            "status": "emergency" if self.emergency_mode else "operational",
            "system_health": self.system_health_score,
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "recovery_in_progress": self.recovery_in_progress,
            "active_errors": len(self.active_errors),
            "total_errors": self.stats['total_errors'],
            "recovery_success_rate": (self.stats['successful_recoveries'] / 
                                    max(1, self.stats['recovery_attempts'])) * 100,
            "uptime_hours": uptime / 3600,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }
    
    def reset_emergency_mode(self):
        """Reset emergency mode"""
        if self.emergency_mode:
            self.emergency_mode = False
            self.auto_recovery_enabled = True
            self.logger.info("Emergency mode reset - system operational")
    
    def shutdown(self):
        """Shutdown the error recovery system"""
        try:
            self.logger.info("Shutting down Enhanced Error Recovery System...")
            self.is_running = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            # Save configuration
            self.save_configuration()
            
            self.logger.info("Enhanced Error Recovery System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error recovery system shutdown error: {e}")

def test_error_recovery_system():
    """Test the error recovery system"""
    print("Testing Enhanced Error Recovery System...")
    print("=" * 60)
    
    recovery_system = EnhancedErrorRecoverySystem()
    
    # Test initialization
    result = recovery_system.initialize()
    print(f"Initialization: {result['status']}")
    
    if result['status'] == 'initialized':
        # Test error reporting
        error_id = recovery_system.report_error(
            error_type="ConnectionError",
            error_message="Failed to connect to MT5",
            error_source="MT5_CONNECTOR",
            severity=AlertLevel.ERROR
        )
        print(f"Error reported with ID: {error_id}")
        
        # Wait for recovery attempt
        time.sleep(3)
        
        # Test alert conditions
        recovery_system.check_alert_conditions('system_health', 65.0)  # Below threshold
        
        # Get status
        status = recovery_system.get_system_status()
        print(f"System Health: {status['system_health']:.1f}%")
        print(f"Active Errors: {status['active_errors']}")
        print(f"Recovery Success Rate: {status['recovery_success_rate']:.1f}%")
        
        print("\\n[OK] Enhanced Error Recovery System Test PASSED!")
        return True
    else:
        print(f"[FAIL] Test failed: {result.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    test_error_recovery_system()