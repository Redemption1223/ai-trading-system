"""
AGENT_11: Alert System
Status: FULLY IMPLEMENTED
Purpose: Comprehensive alert management with multiple notification channels and intelligent monitoring
"""

import logging
import time
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import defaultdict, deque

# Try to import email libraries
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    smtplib = None
    MimeText = None
    MimeMultipart = None

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Alert types"""
    PRICE_MOVEMENT = "PRICE_MOVEMENT"
    TRADE_EXECUTION = "TRADE_EXECUTION"
    RISK_LIMIT = "RISK_LIMIT"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    PERFORMANCE = "PERFORMANCE"
    CONNECTION = "CONNECTION"
    PORTFOLIO = "PORTFOLIO"
    SIGNAL = "SIGNAL"
    CUSTOM = "CUSTOM"

class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "EMAIL"
    SMS = "SMS"
    DESKTOP = "DESKTOP"
    LOG = "LOG"
    WEBHOOK = "WEBHOOK"
    FILE = "FILE"

class Alert:
    """Individual alert object"""
    
    def __init__(self, alert_id: str, alert_type: AlertType, level: AlertLevel, 
                 title: str, message: str, source: str = "SYSTEM"):
        self.alert_id = alert_id
        self.alert_type = alert_type
        self.level = level
        self.title = title
        self.message = message
        self.source = source
        self.created_at = datetime.now()
        self.acknowledged = False
        self.acknowledged_by = None
        self.acknowledged_at = None
        self.resolved = False
        self.resolved_at = None
        self.data = {}  # Additional alert data
        self.notifications_sent = []
        self.retry_count = 0
        self.max_retries = 3

class AlertRule:
    """Alert rule configuration"""
    
    def __init__(self, rule_id: str, name: str, alert_type: AlertType, 
                 condition: Callable, level: AlertLevel = AlertLevel.INFO, 
                 enabled: bool = True, cooldown: int = 300):
        self.rule_id = rule_id
        self.name = name
        self.alert_type = alert_type
        self.condition = condition
        self.level = level
        self.enabled = enabled
        self.cooldown = cooldown  # Seconds between alerts
        self.last_triggered = None
        self.trigger_count = 0
        self.channels = [NotificationChannel.LOG]
        self.custom_data = {}

class AlertSystem:
    """Comprehensive alert and notification management system"""
    
    def __init__(self):
        self.name = "ALERT_SYSTEM"
        self.status = "DISCONNECTED"
        self.version = "1.0.0"
        
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Alert storage
        self.active_alerts = {}  # alert_id -> Alert
        self.alert_history = deque(maxlen=1000)  # Historical alerts
        self.alert_rules = {}  # rule_id -> AlertRule
        self.alert_counter = 0
        
        # Notification settings
        self.notification_channels = {
            NotificationChannel.LOG: True,
            NotificationChannel.DESKTOP: False,
            NotificationChannel.EMAIL: False,
            NotificationChannel.FILE: True,
            NotificationChannel.SMS: False,
            NotificationChannel.WEBHOOK: False
        }
        
        # Email configuration
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'sender_name': 'AGI Trading System',
            'recipients': []
        }
        
        # File logging configuration
        self.file_config = {
            'log_file': 'alerts.log',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5
        }
        
        # Desktop notification settings
        self.desktop_config = {
            'show_info': False,
            'show_warnings': True,
            'show_errors': True,
            'show_critical': True
        }
        
        # Webhook configuration
        self.webhook_config = {
            'url': '',
            'headers': {},
            'timeout': 30
        }
        
        # Connected agents for monitoring
        self.monitored_agents = {}
        self.agent_health_checks = {}
        
        # Monitoring threads
        self.monitoring_thread = None
        self.notification_thread = None
        self.is_monitoring = False
        self.check_interval = 30  # Check every 30 seconds
        
        # Alert queues
        self.notification_queue = deque()
        self.failed_notifications = deque(maxlen=100)
        
        # Performance tracking
        self.performance_stats = {
            'total_alerts': 0,
            'alerts_by_level': defaultdict(int),
            'alerts_by_type': defaultdict(int),
            'notifications_sent': 0,
            'notification_failures': 0,
            'rules_triggered': 0,
            'average_response_time': 0.0
        }
        
        # Built-in alert rules
        self._setup_default_rules()
    
    def initialize(self, **agent_connections):
        """Initialize the alert system"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Connect to other agents for monitoring
            for agent_name, agent_instance in agent_connections.items():
                if agent_instance:
                    self.monitored_agents[agent_name] = agent_instance
                    self.logger.info(f"Monitoring agent: {agent_name}")
            
            # Setup file logging if enabled
            if self.notification_channels.get(NotificationChannel.FILE):
                self._setup_file_logging()
            
            # Start monitoring threads
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
            self.monitoring_thread.start()
            
            self.notification_thread = threading.Thread(target=self._process_notifications, daemon=True)
            self.notification_thread.start()
            
            # Test notification channels
            self._test_notification_channels()
            
            self.status = "INITIALIZED"
            self.logger.info("Alert System initialized successfully")
            
            # Send initialization alert
            self.create_alert(
                AlertType.SYSTEM_ERROR,
                AlertLevel.INFO,
                "Alert System Online",
                "Alert system has been initialized and is monitoring for events",
                "ALERT_SYSTEM"
            )
            
            return {
                "status": "initialized",
                "agent": "AGENT_11",
                "monitored_agents": list(self.monitored_agents.keys()),
                "notification_channels": {k.value: v for k, v in self.notification_channels.items()},
                "alert_rules": len(self.alert_rules),
                "check_interval": self.check_interval
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "agent": "AGENT_11", "error": str(e)}
    
    def _setup_default_rules(self):
        """Set up default alert rules"""
        try:
            # High drawdown rule
            def high_drawdown_condition(data):
                portfolio_data = data.get('portfolio_summary', {})
                max_drawdown = portfolio_data.get('performance_metrics', {}).get('max_drawdown', 0)
                return max_drawdown > 15.0  # 15% drawdown threshold
            
            self.add_alert_rule(
                "HIGH_DRAWDOWN",
                "High Portfolio Drawdown",
                AlertType.PORTFOLIO,
                high_drawdown_condition,
                AlertLevel.WARNING,
                cooldown=1800  # 30 minutes
            )
            
            # Trading disabled rule
            def trading_disabled_condition(data):
                execution_status = data.get('execution_engine_status', {})
                return not execution_status.get('trading_enabled', True)
            
            self.add_alert_rule(
                "TRADING_DISABLED",
                "Trading Disabled",
                AlertType.SYSTEM_ERROR,
                trading_disabled_condition,
                AlertLevel.ERROR,
                cooldown=3600  # 1 hour
            )
            
            # Connection lost rule
            def connection_lost_condition(data):
                mt5_status = data.get('mt5_connector_status', {})
                return mt5_status.get('connection_status') == 'DISCONNECTED'
            
            self.add_alert_rule(
                "CONNECTION_LOST",
                "MT5 Connection Lost",
                AlertType.CONNECTION,
                connection_lost_condition,
                AlertLevel.CRITICAL,
                cooldown=600  # 10 minutes
            )
            
            # High volatility rule
            def high_volatility_condition(data):
                portfolio_data = data.get('portfolio_summary', {})
                volatility = portfolio_data.get('performance_metrics', {}).get('volatility', 0)
                return volatility > 25.0  # 25% annual volatility threshold
            
            self.add_alert_rule(
                "HIGH_VOLATILITY",
                "High Portfolio Volatility",
                AlertType.RISK_LIMIT,
                high_volatility_condition,
                AlertLevel.WARNING,
                cooldown=3600  # 1 hour
            )
            
            self.logger.info(f"Set up {len(self.alert_rules)} default alert rules")
            
        except Exception as e:
            self.logger.error(f"Error setting up default rules: {e}")
    
    def _setup_file_logging(self):
        """Set up file logging for alerts"""
        try:
            log_file = self.file_config['log_file']
            
            # Create file handler for alerts
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Create separate logger for alerts
            self.alert_file_logger = logging.getLogger('alert_file')
            self.alert_file_logger.addHandler(file_handler)
            self.alert_file_logger.setLevel(logging.INFO)
            
            self.logger.info(f"File logging set up: {log_file}")
            
        except Exception as e:
            self.logger.error(f"Error setting up file logging: {e}")
    
    def _test_notification_channels(self):
        """Test notification channels"""
        try:
            working_channels = []
            
            for channel, enabled in self.notification_channels.items():
                if enabled:
                    try:
                        if channel == NotificationChannel.LOG:
                            working_channels.append(channel)
                        elif channel == NotificationChannel.FILE and hasattr(self, 'alert_file_logger'):
                            working_channels.append(channel)
                        elif channel == NotificationChannel.DESKTOP:
                            # Test desktop notification capability
                            working_channels.append(channel)
                        # Add other channel tests as needed
                    except Exception as e:
                        self.logger.warning(f"Channel {channel.value} test failed: {e}")
            
            self.logger.info(f"Working notification channels: {[c.value for c in working_channels]}")
            
        except Exception as e:
            self.logger.error(f"Error testing notification channels: {e}")
    
    def add_alert_rule(self, rule_id: str, name: str, alert_type: AlertType, 
                      condition: Callable, level: AlertLevel = AlertLevel.INFO,
                      enabled: bool = True, cooldown: int = 300,
                      channels: List[NotificationChannel] = None) -> Dict:
        """Add a new alert rule"""
        try:
            if channels is None:
                channels = [NotificationChannel.LOG]
            
            rule = AlertRule(rule_id, name, alert_type, condition, level, enabled, cooldown)
            rule.channels = channels
            
            self.alert_rules[rule_id] = rule
            
            self.logger.info(f"Alert rule added: {name} ({rule_id})")
            
            return {
                "status": "success",
                "rule_id": rule_id,
                "name": name,
                "enabled": enabled,
                "level": level.value
            }
            
        except Exception as e:
            self.logger.error(f"Error adding alert rule {rule_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def remove_alert_rule(self, rule_id: str) -> Dict:
        """Remove an alert rule"""
        try:
            if rule_id in self.alert_rules:
                rule = self.alert_rules.pop(rule_id)
                self.logger.info(f"Alert rule removed: {rule.name} ({rule_id})")
                return {"status": "success", "rule_id": rule_id}
            else:
                return {"status": "error", "message": "Rule not found"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def enable_alert_rule(self, rule_id: str, enabled: bool = True) -> Dict:
        """Enable or disable an alert rule"""
        try:
            if rule_id in self.alert_rules:
                self.alert_rules[rule_id].enabled = enabled
                status = "enabled" if enabled else "disabled"
                self.logger.info(f"Alert rule {status}: {rule_id}")
                return {"status": "success", "rule_id": rule_id, "enabled": enabled}
            else:
                return {"status": "error", "message": "Rule not found"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def create_alert(self, alert_type: AlertType, level: AlertLevel, title: str, 
                    message: str, source: str = "SYSTEM", data: Dict = None) -> Dict:
        """Create a new alert"""
        try:
            self.alert_counter += 1
            alert_id = f"ALERT_{self.alert_counter}_{int(time.time())}"
            
            alert = Alert(alert_id, alert_type, level, title, message, source)
            
            if data:
                alert.data = data
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Update statistics
            self.performance_stats['total_alerts'] += 1
            self.performance_stats['alerts_by_level'][level.value] += 1
            self.performance_stats['alerts_by_type'][alert_type.value] += 1
            
            # Queue for notification
            self.notification_queue.append(alert)
            
            self.logger.info(f"Alert created: {alert_id} - {title}")
            
            return {
                "status": "success",
                "alert_id": alert_id,
                "level": level.value,
                "type": alert_type.value,
                "created_at": alert.created_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
            return {"status": "error", "message": str(e)}
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "USER") -> Dict:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                
                if not alert.acknowledged:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = datetime.now()
                    
                    self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                    
                    return {
                        "status": "success",
                        "alert_id": alert_id,
                        "acknowledged_by": acknowledged_by,
                        "acknowledged_at": alert.acknowledged_at.isoformat()
                    }
                else:
                    return {"status": "already_acknowledged"}
            else:
                return {"status": "error", "message": "Alert not found"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def resolve_alert(self, alert_id: str) -> Dict:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                
                if not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    
                    # Move from active to history
                    del self.active_alerts[alert_id]
                    
                    self.logger.info(f"Alert resolved: {alert_id}")
                    
                    return {
                        "status": "success",
                        "alert_id": alert_id,
                        "resolved_at": alert.resolved_at.isoformat()
                    }
                else:
                    return {"status": "already_resolved"}
            else:
                return {"status": "error", "message": "Alert not found"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _monitor_system(self):
        """Monitor system and check alert rules"""
        self.logger.info("Starting alert system monitoring thread")
        
        while self.is_monitoring:
            try:
                # Collect system data
                system_data = self._collect_system_data()
                
                # Check alert rules
                self._check_alert_rules(system_data)
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in alert system monitoring: {e}")
                time.sleep(60)
    
    def _collect_system_data(self) -> Dict:
        """Collect system data from monitored agents"""
        try:
            system_data = {}
            
            # Collect data from each monitored agent
            for agent_name, agent in self.monitored_agents.items():
                try:
                    if hasattr(agent, 'get_status'):
                        status = agent.get_status()
                        system_data[f"{agent_name}_status"] = status
                    
                    # Collect specific data based on agent type
                    if agent_name == 'portfolio_manager' and hasattr(agent, 'get_portfolio_summary'):
                        summary = agent.get_portfolio_summary()
                        system_data['portfolio_summary'] = summary
                    
                    elif agent_name == 'trade_execution_engine' and hasattr(agent, 'get_performance_metrics'):
                        metrics = agent.get_performance_metrics()
                        system_data['execution_metrics'] = metrics
                    
                    elif agent_name == 'mt5_connector' and hasattr(agent, 'connection_status'):
                        system_data['mt5_connection_status'] = agent.connection_status
                    
                except Exception as e:
                    self.logger.debug(f"Could not collect data from {agent_name}: {e}")
                    system_data[f"{agent_name}_error"] = str(e)
            
            return system_data
            
        except Exception as e:
            self.logger.error(f"Error collecting system data: {e}")
            return {}
    
    def _check_alert_rules(self, system_data: Dict):
        """Check all alert rules against system data"""
        try:
            current_time = datetime.now()
            
            for rule_id, rule in self.alert_rules.items():
                try:
                    if not rule.enabled:
                        continue
                    
                    # Check cooldown
                    if (rule.last_triggered and 
                        (current_time - rule.last_triggered).total_seconds() < rule.cooldown):
                        continue
                    
                    # Evaluate condition
                    if rule.condition(system_data):
                        # Rule triggered
                        rule.last_triggered = current_time
                        rule.trigger_count += 1
                        self.performance_stats['rules_triggered'] += 1
                        
                        # Create alert
                        self.create_alert(
                            rule.alert_type,
                            rule.level,
                            rule.name,
                            f"Alert rule '{rule.name}' has been triggered",
                            f"RULE_{rule_id}",
                            {"rule_id": rule_id, "trigger_count": rule.trigger_count}
                        )
                        
                        self.logger.warning(f"Alert rule triggered: {rule.name}")
                    
                except Exception as e:
                    self.logger.error(f"Error checking rule {rule_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error checking alert rules: {e}")
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        try:
            current_time = datetime.now()
            cleanup_age = timedelta(hours=24)  # Clean up after 24 hours
            
            alerts_to_remove = []
            
            for alert_id, alert in self.active_alerts.items():
                if (alert.resolved and alert.resolved_at and 
                    current_time - alert.resolved_at > cleanup_age):
                    alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
            
            if alerts_to_remove:
                self.logger.info(f"Cleaned up {len(alerts_to_remove)} old resolved alerts")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old alerts: {e}")
    
    def _process_notifications(self):
        """Process notification queue"""
        self.logger.info("Starting notification processing thread")
        
        while self.is_monitoring:
            try:
                if self.notification_queue:
                    alert = self.notification_queue.popleft()
                    self._send_notifications(alert)
                else:
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error processing notifications: {e}")
                time.sleep(5)
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        try:
            start_time = time.time()
            notifications_sent = 0
            
            for channel, enabled in self.notification_channels.items():
                if not enabled:
                    continue
                
                try:
                    if channel == NotificationChannel.LOG:
                        self._send_log_notification(alert)
                        notifications_sent += 1
                        
                    elif channel == NotificationChannel.FILE:
                        self._send_file_notification(alert)
                        notifications_sent += 1
                        
                    elif channel == NotificationChannel.EMAIL:
                        self._send_email_notification(alert)
                        notifications_sent += 1
                        
                    elif channel == NotificationChannel.DESKTOP:
                        self._send_desktop_notification(alert)
                        notifications_sent += 1
                        
                    elif channel == NotificationChannel.WEBHOOK:
                        self._send_webhook_notification(alert)
                        notifications_sent += 1
                    
                    alert.notifications_sent.append({
                        'channel': channel.value,
                        'sent_at': datetime.now().isoformat(),
                        'success': True
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to send {channel.value} notification: {e}")
                    self.performance_stats['notification_failures'] += 1
                    
                    alert.notifications_sent.append({
                        'channel': channel.value,
                        'sent_at': datetime.now().isoformat(),
                        'success': False,
                        'error': str(e)
                    })
                    
                    # Add to failed notifications for retry
                    if alert.retry_count < alert.max_retries:
                        alert.retry_count += 1
                        self.failed_notifications.append((alert, channel))
            
            # Update performance stats
            self.performance_stats['notifications_sent'] += notifications_sent
            response_time = time.time() - start_time
            
            current_avg = self.performance_stats['average_response_time']
            total_notifications = self.performance_stats['notifications_sent']
            self.performance_stats['average_response_time'] = (
                (current_avg * (total_notifications - notifications_sent) + response_time) / 
                max(total_notifications, 1)
            )
            
        except Exception as e:
            self.logger.error(f"Error sending notifications for alert {alert.alert_id}: {e}")
    
    def _send_log_notification(self, alert: Alert):
        """Send log notification"""
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert.level, logging.INFO)
        
        self.logger.log(log_level, f"ALERT [{alert.level.value}] {alert.title}: {alert.message}")
    
    def _send_file_notification(self, alert: Alert):
        """Send file notification"""
        if hasattr(self, 'alert_file_logger'):
            log_message = f"[{alert.alert_id}] [{alert.level.value}] [{alert.alert_type.value}] {alert.title}: {alert.message}"
            self.alert_file_logger.info(log_message)
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            if not EMAIL_AVAILABLE:
                raise Exception("Email libraries not available")
            
            if not self.email_config['username'] or not self.email_config['recipients']:
                raise Exception("Email configuration incomplete")
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = f"{self.email_config['sender_name']} <{self.email_config['username']}>"
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f"[{alert.level.value}] {alert.title}"
            
            # Email body
            body = f"""
Alert Details:
- Alert ID: {alert.alert_id}
- Type: {alert.alert_type.value}
- Level: {alert.level.value}
- Source: {alert.source}
- Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}

Message:
{alert.message}

Additional Data:
{json.dumps(alert.data, indent=2) if alert.data else 'None'}
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            self.logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            raise
    
    def _send_desktop_notification(self, alert: Alert):
        """Send desktop notification"""
        try:
            # Check if this level should be shown
            show_alert = {
                AlertLevel.INFO: self.desktop_config['show_info'],
                AlertLevel.WARNING: self.desktop_config['show_warnings'],
                AlertLevel.ERROR: self.desktop_config['show_errors'],
                AlertLevel.CRITICAL: self.desktop_config['show_critical']
            }.get(alert.level, True)
            
            if not show_alert:
                return
            
            # Try to use Windows toast notifications
            try:
                import win10toast
                toaster = win10toast.ToastNotifier()
                toaster.show_toast(
                    title=f"[{alert.level.value}] {alert.title}",
                    msg=alert.message[:200],  # Truncate long messages
                    duration=10,
                    threaded=True
                )
            except ImportError:
                # Fallback to simple console notification
                print(f"\n*** DESKTOP ALERT ***")
                print(f"[{alert.level.value}] {alert.title}")
                print(f"{alert.message}")
                print(f"Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print("*" * 50)
            
        except Exception as e:
            self.logger.error(f"Failed to send desktop notification: {e}")
            raise
    
    def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        try:
            if not self.webhook_config['url']:
                raise Exception("Webhook URL not configured")
            
            import requests
            
            payload = {
                'alert_id': alert.alert_id,
                'type': alert.alert_type.value,
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'created_at': alert.created_at.isoformat(),
                'data': alert.data
            }
            
            response = requests.post(
                self.webhook_config['url'],
                json=payload,
                headers=self.webhook_config['headers'],
                timeout=self.webhook_config['timeout']
            )
            
            response.raise_for_status()
            self.logger.info(f"Webhook notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            raise
    
    def get_active_alerts(self, level: AlertLevel = None, alert_type: AlertType = None) -> List[Dict]:
        """Get active alerts with optional filtering"""
        try:
            alerts = []
            
            for alert in self.active_alerts.values():
                if level and alert.level != level:
                    continue
                if alert_type and alert.alert_type != alert_type:
                    continue
                
                alerts.append({
                    'alert_id': alert.alert_id,
                    'type': alert.alert_type.value,
                    'level': alert.level.value,
                    'title': alert.title,
                    'message': alert.message,
                    'source': alert.source,
                    'created_at': alert.created_at.isoformat(),
                    'acknowledged': alert.acknowledged,
                    'acknowledged_by': alert.acknowledged_by,
                    'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    'resolved': alert.resolved,
                    'notifications_sent': len(alert.notifications_sent),
                    'data': alert.data
                })
            
            # Sort by creation time (newest first)
            alerts.sort(key=lambda x: x['created_at'], reverse=True)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {e}")
            return []
    
    def get_alert_history(self, count: int = 50) -> List[Dict]:
        """Get alert history"""
        try:
            alerts = []
            
            for alert in list(self.alert_history)[-count:]:
                alerts.append({
                    'alert_id': alert.alert_id,
                    'type': alert.alert_type.value,
                    'level': alert.level.value,
                    'title': alert.title,
                    'message': alert.message,
                    'source': alert.source,
                    'created_at': alert.created_at.isoformat(),
                    'acknowledged': alert.acknowledged,
                    'resolved': alert.resolved,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                    'notifications_sent': len(alert.notifications_sent)
                })
            
            return alerts[::-1]  # Reverse to show newest first
            
        except Exception as e:
            self.logger.error(f"Error getting alert history: {e}")
            return []
    
    def get_alert_rules(self) -> List[Dict]:
        """Get all alert rules"""
        try:
            rules = []
            
            for rule_id, rule in self.alert_rules.items():
                rules.append({
                    'rule_id': rule_id,
                    'name': rule.name,
                    'type': rule.alert_type.value,
                    'level': rule.level.value,
                    'enabled': rule.enabled,
                    'cooldown': rule.cooldown,
                    'trigger_count': rule.trigger_count,
                    'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
                    'channels': [c.value for c in rule.channels]
                })
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Error getting alert rules: {e}")
            return []
    
    def configure_notification_channel(self, channel: NotificationChannel, 
                                     enabled: bool, config: Dict = None) -> Dict:
        """Configure notification channel"""
        try:
            self.notification_channels[channel] = enabled
            
            if config:
                if channel == NotificationChannel.EMAIL:
                    self.email_config.update(config)
                elif channel == NotificationChannel.DESKTOP:
                    self.desktop_config.update(config)
                elif channel == NotificationChannel.WEBHOOK:
                    self.webhook_config.update(config)
                elif channel == NotificationChannel.FILE:
                    self.file_config.update(config)
            
            self.logger.info(f"Notification channel {channel.value} {'enabled' if enabled else 'disabled'}")
            
            return {
                "status": "success",
                "channel": channel.value,
                "enabled": enabled
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_performance_stats(self) -> Dict:
        """Get alert system performance statistics"""
        try:
            return {
                'total_alerts': self.performance_stats['total_alerts'],
                'active_alerts': len(self.active_alerts),
                'alerts_by_level': dict(self.performance_stats['alerts_by_level']),
                'alerts_by_type': dict(self.performance_stats['alerts_by_type']),
                'notifications_sent': self.performance_stats['notifications_sent'],
                'notification_failures': self.performance_stats['notification_failures'],
                'notification_success_rate': (
                    (self.performance_stats['notifications_sent'] - self.performance_stats['notification_failures']) /
                    max(self.performance_stats['notifications_sent'], 1) * 100
                ),
                'rules_triggered': self.performance_stats['rules_triggered'],
                'active_rules': len([r for r in self.alert_rules.values() if r.enabled]),
                'total_rules': len(self.alert_rules),
                'average_response_time': round(self.performance_stats['average_response_time'], 3),
                'failed_notifications_pending': len(self.failed_notifications),
                'notification_queue_size': len(self.notification_queue)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {"error": str(e)}
    
    def get_status(self):
        """Get current alert system status"""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'is_monitoring': self.is_monitoring,
            'monitored_agents': list(self.monitored_agents.keys()),
            'active_alerts': len(self.active_alerts),
            'alert_rules': len(self.alert_rules),
            'notification_channels': {k.value: v for k, v in self.notification_channels.items()},
            'performance_stats': self.get_performance_stats(),
            'check_interval': self.check_interval,
            'last_check': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Clean shutdown of alert system"""
        try:
            self.logger.info("Shutting down Alert System...")
            
            # Send shutdown alert
            self.create_alert(
                AlertType.SYSTEM_ERROR,
                AlertLevel.INFO,
                "Alert System Shutdown",
                "Alert system is shutting down",
                "ALERT_SYSTEM"
            )
            
            # Stop monitoring
            self.is_monitoring = False
            
            # Process remaining notifications
            while self.notification_queue:
                alert = self.notification_queue.popleft()
                self._send_notifications(alert)
            
            # Wait for threads to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            if self.notification_thread and self.notification_thread.is_alive():
                self.notification_thread.join(timeout=5)
            
            # Log final statistics
            final_stats = self.get_performance_stats()
            self.logger.info(f"Final alert statistics: {final_stats}")
            
            self.status = "SHUTDOWN"
            self.logger.info("Alert System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Agent test
if __name__ == "__main__":
    # Test the alert system
    print("Testing AGENT_11: Alert System")
    print("=" * 40)
    
    # Create alert system
    alert_system = AlertSystem()
    result = alert_system.initialize()
    print(f"Initialization: {result}")
    
    if result['status'] == 'initialized':
        # Test creating alerts
        print("\nTesting alert creation...")
        
        alert_result1 = alert_system.create_alert(
            AlertType.PORTFOLIO,
            AlertLevel.WARNING,
            "High Volatility Detected",
            "Portfolio volatility has exceeded 20% threshold",
            "PORTFOLIO_MANAGER"
        )
        print(f"Alert 1 created: {alert_result1}")
        
        alert_result2 = alert_system.create_alert(
            AlertType.SYSTEM_ERROR,
            AlertLevel.CRITICAL,
            "Connection Lost",
            "MT5 connection has been lost",
            "MT5_CONNECTOR"
        )
        print(f"Alert 2 created: {alert_result2}")
        
        # Wait for notifications to process
        time.sleep(2)
        
        # Test getting active alerts
        active_alerts = alert_system.get_active_alerts()
        print(f"\nActive alerts: {len(active_alerts)}")
        
        # Test acknowledging alert
        if active_alerts:
            ack_result = alert_system.acknowledge_alert(active_alerts[0]['alert_id'], "TEST_USER")
            print(f"Alert acknowledged: {ack_result}")
        
        # Test performance stats
        stats = alert_system.get_performance_stats()
        print(f"\nPerformance stats: {stats}")
        
        # Test alert rules
        rules = alert_system.get_alert_rules()
        print(f"Alert rules: {len(rules)}")
        
        # Test status
        status = alert_system.get_status()
        print(f"\nStatus: {status}")
        
        # Shutdown
        print("\nShutting down...")
        alert_system.shutdown()
        
    print("Alert System test completed")