"""
AGENT_12: Configuration Manager
Status: FULLY IMPLEMENTED
Purpose: Centralized configuration management with validation, persistence, and hot-reloading
"""

import logging
import json
import os
import time
import threading
from datetime import datetime, timedelta

# Graceful YAML handling
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from collections import defaultdict
import shutil
import hashlib

class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    INI = "ini"

class ConfigScope(Enum):
    """Configuration scopes"""
    SYSTEM = "system"
    AGENT = "agent"
    USER = "user"
    RUNTIME = "runtime"

class ConfigurationManager:
    """Centralized configuration management system"""
    
    def __init__(self, config_dir: str = "config"):
        self.name = "CONFIGURATION_MANAGER"
        self.status = "DISCONNECTED"
        self.version = "1.0.0"
        
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Configuration directory
        self.config_dir = config_dir
        self.backup_dir = os.path.join(config_dir, "backups")
        self.templates_dir = os.path.join(config_dir, "templates")
        
        # Configuration storage
        self.configurations = {}  # scope -> config_name -> config_data
        self.config_metadata = {}  # config_name -> metadata
        self.config_watchers = {}  # config_name -> list of callbacks
        self.config_validators = {}  # config_name -> validator function
        
        # File monitoring
        self.file_hashes = {}  # file_path -> hash
        self.monitoring_thread = None
        self.is_monitoring = False
        self.check_interval = 5  # Check for changes every 5 seconds
        
        # Configuration schemas
        self.schemas = {}
        self.default_configs = {}
        
        # Change tracking
        self.change_history = []
        self.max_history = 100
        self.pending_changes = {}
        
        # Performance tracking
        self.performance_stats = {
            'configs_loaded': 0,
            'configs_saved': 0,
            'validation_errors': 0,
            'hot_reloads': 0,
            'backups_created': 0,
            'config_changes': 0
        }
        
        # Initialize default configurations
        self._setup_default_configurations()
    
    def initialize(self):
        """Initialize the configuration manager"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Create directory structure
            self._create_directory_structure()
            
            # Load existing configurations
            self._load_existing_configurations()
            
            # Set up configuration schemas
            self._setup_configuration_schemas()
            
            # Start file monitoring
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitor_config_files, daemon=True)
            self.monitoring_thread.start()
            
            # Validate all loaded configurations
            self._validate_all_configurations()
            
            self.status = "INITIALIZED"
            self.logger.info("Configuration Manager initialized successfully")
            
            return {
                "status": "initialized",
                "agent": "AGENT_12",
                "config_dir": self.config_dir,
                "configurations_loaded": len(self.configurations),
                "schemas_available": len(self.schemas),
                "monitoring_enabled": self.is_monitoring,
                "check_interval": self.check_interval
            }
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.status = "FAILED"
            return {"status": "failed", "agent": "AGENT_12", "error": str(e)}
    
    def _create_directory_structure(self):
        """Create necessary directory structure"""
        try:
            directories = [
                self.config_dir,
                self.backup_dir,
                self.templates_dir,
                os.path.join(self.config_dir, "system"),
                os.path.join(self.config_dir, "agents"),
                os.path.join(self.config_dir, "users"),
                os.path.join(self.config_dir, "runtime")
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            self.logger.info("Configuration directory structure created")
            
        except Exception as e:
            self.logger.error(f"Error creating directory structure: {e}")
            raise
    
    def _setup_default_configurations(self):
        """Set up default configurations for all agents"""
        try:
            # System configuration
            self.default_configs['system'] = {
                'general': {
                    'system_name': 'AGI Trading System',
                    'version': '1.0.0',
                    'timezone': 'UTC',
                    'log_level': 'INFO',
                    'debug_mode': False,
                    'max_memory_usage': '2GB',
                    'max_cpu_usage': 80
                },
                'security': {
                    'encryption_enabled': True,
                    'session_timeout': 3600,
                    'max_login_attempts': 3,
                    'password_policy': {
                        'min_length': 8,
                        'require_special_chars': True,
                        'require_numbers': True
                    }
                }
            }
            
            # Agent configurations
            self.default_configs['agents'] = {
                'mt5_connector': {
                    'connection_timeout': 30,
                    'retry_attempts': 3,
                    'retry_delay': 5,
                    'auto_reconnect': True,
                    'heartbeat_interval': 60
                },
                'signal_coordinator': {
                    'signal_timeout': 30,
                    'max_concurrent_signals': 10,
                    'signal_quality_threshold': 0.7,
                    'coordination_interval': 1
                },
                'risk_calculator': {
                    'default_risk_model': 'kelly_criterion',
                    'max_position_size': 0.1,
                    'max_portfolio_risk': 0.02,
                    'stop_loss_multiplier': 2.0,
                    'take_profit_multiplier': 3.0
                },
                'portfolio_manager': {
                    'rebalance_frequency': 'daily',
                    'rebalance_threshold': 0.05,
                    'max_correlation_exposure': 0.3,
                    'risk_level': 'moderate'
                },
                'alert_system': {
                    'check_interval': 30,
                    'max_alerts': 1000,
                    'notification_channels': ['log', 'file'],
                    'alert_retention_days': 30
                }
            }
            
            # User configuration template
            self.default_configs['user_template'] = {
                'preferences': {
                    'theme': 'dark',
                    'language': 'en',
                    'timezone': 'UTC',
                    'dashboard_layout': 'default'
                },
                'trading': {
                    'default_symbol': 'EURUSD',
                    'default_timeframe': 'H1',
                    'auto_trading': False,
                    'risk_tolerance': 'medium'
                },
                'notifications': {
                    'email_enabled': False,
                    'desktop_enabled': True,
                    'sound_enabled': True,
                    'alert_levels': ['WARNING', 'ERROR', 'CRITICAL']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up default configurations: {e}")
    
    def _setup_configuration_schemas(self):
        """Set up configuration validation schemas"""
        try:
            # System schema
            self.schemas['system'] = {
                'general': {
                    'system_name': {'type': 'string', 'required': True},
                    'version': {'type': 'string', 'required': True},
                    'timezone': {'type': 'string', 'required': True},
                    'log_level': {'type': 'string', 'enum': ['DEBUG', 'INFO', 'WARNING', 'ERROR']},
                    'debug_mode': {'type': 'boolean'},
                    'max_memory_usage': {'type': 'string'},
                    'max_cpu_usage': {'type': 'number', 'minimum': 1, 'maximum': 100}
                }
            }
            
            # Agent schemas
            self.schemas['mt5_connector'] = {
                'connection_timeout': {'type': 'number', 'minimum': 1, 'maximum': 300},
                'retry_attempts': {'type': 'number', 'minimum': 1, 'maximum': 10},
                'retry_delay': {'type': 'number', 'minimum': 1, 'maximum': 60},
                'auto_reconnect': {'type': 'boolean'},
                'heartbeat_interval': {'type': 'number', 'minimum': 10, 'maximum': 300}
            }
            
            self.schemas['risk_calculator'] = {
                'default_risk_model': {'type': 'string', 'enum': ['fixed_percent', 'kelly_criterion', 'volatility_adjusted']},
                'max_position_size': {'type': 'number', 'minimum': 0.01, 'maximum': 1.0},
                'max_portfolio_risk': {'type': 'number', 'minimum': 0.001, 'maximum': 0.1},
                'stop_loss_multiplier': {'type': 'number', 'minimum': 1.0, 'maximum': 10.0},
                'take_profit_multiplier': {'type': 'number', 'minimum': 1.0, 'maximum': 20.0}
            }
            
            self.logger.info(f"Configuration schemas set up for {len(self.schemas)} configurations")
            
        except Exception as e:
            self.logger.error(f"Error setting up schemas: {e}")
    
    def _load_existing_configurations(self):
        """Load existing configuration files"""
        try:
            configs_loaded = 0
            
            for scope in ConfigScope:
                scope_dir = os.path.join(self.config_dir, scope.value)
                if os.path.exists(scope_dir):
                    for file_name in os.listdir(scope_dir):
                        if file_name.endswith(('.json', '.yaml', '.yml')):
                            config_name = os.path.splitext(file_name)[0]
                            file_path = os.path.join(scope_dir, file_name)
                            
                            try:
                                config_data = self._load_config_file(file_path)
                                self._store_configuration(scope.value, config_name, config_data, file_path)
                                configs_loaded += 1
                                
                            except Exception as e:
                                self.logger.error(f"Failed to load config {file_path}: {e}")
            
            self.performance_stats['configs_loaded'] = configs_loaded
            self.logger.info(f"Loaded {configs_loaded} existing configurations")
            
        except Exception as e:
            self.logger.error(f"Error loading existing configurations: {e}")
    
    def _load_config_file(self, file_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                elif file_path.endswith(('.yaml', '.yml')):
                    if YAML_AVAILABLE:
                        try:
                            return yaml.safe_load(f) or {}
                        except:
                            # Fallback to JSON if YAML fails
                            f.seek(0)
                            return json.load(f)
                    else:
                        # YAML not available, try JSON
                        return json.load(f)
                else:
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Error loading config file {file_path}: {e}")
            raise
    
    def _store_configuration(self, scope: str, config_name: str, config_data: Dict, file_path: str = None):
        """Store configuration in memory"""
        try:
            if scope not in self.configurations:
                self.configurations[scope] = {}
            
            self.configurations[scope][config_name] = config_data
            
            # Store metadata
            self.config_metadata[config_name] = {
                'scope': scope,
                'file_path': file_path,
                'loaded_at': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat(),
                'format': self._detect_format(file_path) if file_path else ConfigFormat.JSON,
                'size': len(str(config_data)),
                'checksum': self._calculate_checksum(config_data)
            }
            
            # Update file hash for monitoring
            if file_path and os.path.exists(file_path):
                self.file_hashes[file_path] = self._calculate_file_hash(file_path)
            
        except Exception as e:
            self.logger.error(f"Error storing configuration {config_name}: {e}")
    
    def _detect_format(self, file_path: str) -> ConfigFormat:
        """Detect configuration file format"""
        if not file_path:
            return ConfigFormat.JSON
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.json':
            return ConfigFormat.JSON
        elif ext in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        else:
            return ConfigFormat.JSON
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for configuration data"""
        try:
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(data_str.encode()).hexdigest()
        except:
            return ""
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def get_configuration(self, scope: str, config_name: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            if scope in self.configurations and config_name in self.configurations[scope]:
                return self.configurations[scope][config_name]
            
            # Check if default configuration exists
            if scope in self.default_configs and config_name in self.default_configs[scope]:
                return self.default_configs[scope][config_name]
            
            return default
            
        except Exception as e:
            self.logger.error(f"Error getting configuration {scope}.{config_name}: {e}")
            return default
    
    def set_configuration(self, scope: str, config_name: str, config_data: Dict, 
                         save_to_file: bool = True, create_backup: bool = True) -> Dict:
        """Set configuration value"""
        try:
            # Validate configuration if schema exists
            if config_name in self.schemas:
                validation_result = self._validate_configuration(config_name, config_data)
                if not validation_result['valid']:
                    return {
                        "status": "error",
                        "message": f"Validation failed: {validation_result['errors']}"
                    }
            
            # Create backup if requested and configuration exists
            if create_backup and scope in self.configurations and config_name in self.configurations[scope]:
                self._create_backup(scope, config_name)
            
            # Store configuration
            old_data = self.configurations.get(scope, {}).get(config_name)
            self._store_configuration(scope, config_name, config_data)
            
            # Save to file if requested
            if save_to_file:
                file_path = self._get_config_file_path(scope, config_name)
                self._save_config_file(file_path, config_data)
                self.performance_stats['configs_saved'] += 1
            
            # Track change
            self._track_configuration_change(scope, config_name, old_data, config_data)
            
            # Notify watchers
            self._notify_configuration_watchers(config_name, config_data)
            
            self.performance_stats['config_changes'] += 1
            
            self.logger.info(f"Configuration updated: {scope}.{config_name}")
            
            return {
                "status": "success",
                "scope": scope,
                "config_name": config_name,
                "saved_to_file": save_to_file
            }
            
        except Exception as e:
            self.logger.error(f"Error setting configuration {scope}.{config_name}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_config_file_path(self, scope: str, config_name: str, format: ConfigFormat = ConfigFormat.JSON) -> str:
        """Get file path for configuration"""
        scope_dir = os.path.join(self.config_dir, scope)
        os.makedirs(scope_dir, exist_ok=True)
        
        extension = '.json' if format == ConfigFormat.JSON else '.yaml'
        return os.path.join(scope_dir, f"{config_name}{extension}")
    
    def _save_config_file(self, file_path: str, config_data: Dict):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, default=str)
            elif file_path.endswith(('.yaml', '.yml')):
                if YAML_AVAILABLE:
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            yaml.safe_dump(config_data, f, default_flow_style=False)
                    except:
                        # Fallback to JSON
                        json_path = file_path.rsplit('.', 1)[0] + '.json'
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(config_data, f, indent=2, default=str)
                else:
                    # YAML not available, save as JSON
                    json_path = file_path.rsplit('.', 1)[0] + '.json'
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(config_data, f, indent=2, default=str)
            
            # Update file hash
            self.file_hashes[file_path] = self._calculate_file_hash(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving config file {file_path}: {e}")
            raise
    
    def _validate_configuration(self, config_name: str, config_data: Dict) -> Dict:
        """Validate configuration against schema"""
        try:
            if config_name not in self.schemas:
                return {"valid": True, "errors": []}
            
            schema = self.schemas[config_name]
            errors = []
            
            # Simple validation implementation
            for field, rules in schema.items():
                value = config_data.get(field)
                
                # Check required fields
                if rules.get('required', False) and value is None:
                    errors.append(f"Required field '{field}' is missing")
                    continue
                
                if value is not None:
                    # Type validation
                    expected_type = rules.get('type')
                    if expected_type == 'string' and not isinstance(value, str):
                        errors.append(f"Field '{field}' must be a string")
                    elif expected_type == 'number' and not isinstance(value, (int, float)):
                        errors.append(f"Field '{field}' must be a number")
                    elif expected_type == 'boolean' and not isinstance(value, bool):
                        errors.append(f"Field '{field}' must be a boolean")
                    
                    # Enum validation
                    if 'enum' in rules and value not in rules['enum']:
                        errors.append(f"Field '{field}' must be one of {rules['enum']}")
                    
                    # Range validation
                    if isinstance(value, (int, float)):
                        if 'minimum' in rules and value < rules['minimum']:
                            errors.append(f"Field '{field}' must be >= {rules['minimum']}")
                        if 'maximum' in rules and value > rules['maximum']:
                            errors.append(f"Field '{field}' must be <= {rules['maximum']}")
            
            if errors:
                self.performance_stats['validation_errors'] += 1
            
            return {"valid": len(errors) == 0, "errors": errors}
            
        except Exception as e:
            self.logger.error(f"Error validating configuration {config_name}: {e}")
            return {"valid": False, "errors": [str(e)]}
    
    def _validate_all_configurations(self):
        """Validate all loaded configurations"""
        try:
            validation_results = {}
            
            for scope, configs in self.configurations.items():
                for config_name, config_data in configs.items():
                    if config_name in self.schemas:
                        result = self._validate_configuration(config_name, config_data)
                        if not result['valid']:
                            validation_results[f"{scope}.{config_name}"] = result['errors']
                            self.logger.warning(f"Configuration validation failed: {scope}.{config_name}: {result['errors']}")
            
            if validation_results:
                self.logger.warning(f"Found {len(validation_results)} configurations with validation errors")
            else:
                self.logger.info("All configurations passed validation")
            
        except Exception as e:
            self.logger.error(f"Error validating configurations: {e}")
    
    def _create_backup(self, scope: str, config_name: str):
        """Create backup of configuration"""
        try:
            if scope not in self.configurations or config_name not in self.configurations[scope]:
                return
            
            config_data = self.configurations[scope][config_name]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{config_name}_{timestamp}.json"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            self.performance_stats['backups_created'] += 1
            self.logger.info(f"Backup created: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating backup for {scope}.{config_name}: {e}")
    
    def _track_configuration_change(self, scope: str, config_name: str, old_data: Any, new_data: Any):
        """Track configuration changes"""
        try:
            change_record = {
                'timestamp': datetime.now().isoformat(),
                'scope': scope,
                'config_name': config_name,
                'old_checksum': self._calculate_checksum(old_data) if old_data else None,
                'new_checksum': self._calculate_checksum(new_data),
                'change_type': 'update' if old_data else 'create'
            }
            
            self.change_history.append(change_record)
            
            # Limit history size
            if len(self.change_history) > self.max_history:
                self.change_history = self.change_history[-self.max_history:]
            
        except Exception as e:
            self.logger.error(f"Error tracking configuration change: {e}")
    
    def add_configuration_watcher(self, config_name: str, callback: callable) -> Dict:
        """Add watcher for configuration changes"""
        try:
            if config_name not in self.config_watchers:
                self.config_watchers[config_name] = []
            
            self.config_watchers[config_name].append(callback)
            
            self.logger.info(f"Configuration watcher added for {config_name}")
            
            return {
                "status": "success",
                "config_name": config_name,
                "watchers_count": len(self.config_watchers[config_name])
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def remove_configuration_watcher(self, config_name: str, callback: callable) -> Dict:
        """Remove configuration watcher"""
        try:
            if config_name in self.config_watchers and callback in self.config_watchers[config_name]:
                self.config_watchers[config_name].remove(callback)
                
                if not self.config_watchers[config_name]:
                    del self.config_watchers[config_name]
                
                return {"status": "success"}
            else:
                return {"status": "not_found"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _notify_configuration_watchers(self, config_name: str, config_data: Any):
        """Notify watchers of configuration changes"""
        try:
            if config_name in self.config_watchers:
                for callback in self.config_watchers[config_name]:
                    try:
                        callback(config_name, config_data)
                    except Exception as e:
                        self.logger.error(f"Error in configuration watcher callback: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error notifying configuration watchers: {e}")
    
    def _monitor_config_files(self):
        """Monitor configuration files for changes"""
        self.logger.info("Starting configuration file monitoring")
        
        while self.is_monitoring:
            try:
                # Check for file changes
                for file_path, old_hash in list(self.file_hashes.items()):
                    if os.path.exists(file_path):
                        current_hash = self._calculate_file_hash(file_path)
                        
                        if current_hash != old_hash:
                            # File changed, reload configuration
                            self._handle_file_change(file_path)
                            self.file_hashes[file_path] = current_hash
                            self.performance_stats['hot_reloads'] += 1
                    else:
                        # File deleted
                        del self.file_hashes[file_path]
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error monitoring configuration files: {e}")
                time.sleep(30)
    
    def _handle_file_change(self, file_path: str):
        """Handle configuration file change"""
        try:
            # Determine scope and config name from file path
            rel_path = os.path.relpath(file_path, self.config_dir)
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) >= 2:
                scope = path_parts[0]
                config_name = os.path.splitext(path_parts[1])[0]
                
                # Reload configuration
                config_data = self._load_config_file(file_path)
                old_data = self.configurations.get(scope, {}).get(config_name)
                
                self._store_configuration(scope, config_name, config_data, file_path)
                
                # Track change
                self._track_configuration_change(scope, config_name, old_data, config_data)
                
                # Notify watchers
                self._notify_configuration_watchers(config_name, config_data)
                
                self.logger.info(f"Configuration hot-reloaded: {scope}.{config_name}")
            
        except Exception as e:
            self.logger.error(f"Error handling file change {file_path}: {e}")
    
    def export_configuration(self, scope: str = None, config_name: str = None, 
                           format: ConfigFormat = ConfigFormat.JSON) -> Dict:
        """Export configuration(s)"""
        try:
            if scope and config_name:
                # Export specific configuration
                config_data = self.get_configuration(scope, config_name)
                if config_data is None:
                    return {"status": "error", "message": "Configuration not found"}
                
                return {
                    "status": "success",
                    "scope": scope,
                    "config_name": config_name,
                    "data": config_data,
                    "format": format.value
                }
            
            elif scope:
                # Export all configurations in scope
                if scope not in self.configurations:
                    return {"status": "error", "message": "Scope not found"}
                
                return {
                    "status": "success",
                    "scope": scope,
                    "data": self.configurations[scope],
                    "format": format.value
                }
            
            else:
                # Export all configurations
                return {
                    "status": "success",
                    "data": self.configurations,
                    "format": format.value
                }
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def import_configuration(self, data: Dict, scope: str, config_name: str = None,
                           validate: bool = True, create_backup: bool = True) -> Dict:
        """Import configuration data"""
        try:
            if config_name:
                # Import specific configuration
                if validate and config_name in self.schemas:
                    validation_result = self._validate_configuration(config_name, data)
                    if not validation_result['valid']:
                        return {
                            "status": "error",
                            "message": f"Validation failed: {validation_result['errors']}"
                        }
                
                return self.set_configuration(scope, config_name, data, 
                                            save_to_file=True, create_backup=create_backup)
            
            else:
                # Import multiple configurations
                results = {}
                for name, config_data in data.items():
                    result = self.set_configuration(scope, name, config_data,
                                                  save_to_file=True, create_backup=create_backup)
                    results[name] = result
                
                return {"status": "success", "results": results}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_configuration_metadata(self, config_name: str = None) -> Dict:
        """Get configuration metadata"""
        try:
            if config_name:
                if config_name in self.config_metadata:
                    return self.config_metadata[config_name]
                else:
                    return {"error": "Configuration not found"}
            else:
                return self.config_metadata
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_change_history(self, limit: int = 50) -> List[Dict]:
        """Get configuration change history"""
        try:
            return list(reversed(self.change_history[-limit:]))
        except Exception as e:
            self.logger.error(f"Error getting change history: {e}")
            return []
    
    def get_available_configurations(self) -> Dict:
        """Get list of available configurations"""
        try:
            result = {}
            
            for scope, configs in self.configurations.items():
                result[scope] = []
                for config_name in configs.keys():
                    metadata = self.config_metadata.get(config_name, {})
                    result[scope].append({
                        'name': config_name,
                        'size': metadata.get('size', 0),
                        'last_modified': metadata.get('last_modified'),
                        'format': metadata.get('format', {}).get('value', 'unknown') if isinstance(metadata.get('format'), dict) else str(metadata.get('format', 'unknown'))
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting available configurations: {e}")
            return {}
    
    def reset_configuration(self, scope: str, config_name: str, create_backup: bool = True) -> Dict:
        """Reset configuration to default"""
        try:
            if scope in self.default_configs and config_name in self.default_configs[scope]:
                default_data = self.default_configs[scope][config_name]
                return self.set_configuration(scope, config_name, default_data,
                                            save_to_file=True, create_backup=create_backup)
            else:
                return {"status": "error", "message": "No default configuration available"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_performance_stats(self) -> Dict:
        """Get configuration manager performance statistics"""
        try:
            return {
                'configurations_loaded': len([config for scope in self.configurations.values() 
                                           for config in scope.keys()]),
                'scopes_active': len(self.configurations),
                'watchers_active': len(self.config_watchers),
                'schemas_available': len(self.schemas),
                'monitoring_active': self.is_monitoring,
                'performance_stats': self.performance_stats.copy(),
                'change_history_size': len(self.change_history),
                'backup_directory': self.backup_dir,
                'config_directory': self.config_dir
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_status(self):
        """Get current configuration manager status"""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'config_dir': self.config_dir,
            'monitoring_active': self.is_monitoring,
            'configurations_loaded': len([config for scope in self.configurations.values() 
                                        for config in scope.keys()]),
            'scopes': list(self.configurations.keys()),
            'schemas_available': len(self.schemas),
            'watchers_active': len(self.config_watchers),
            'performance_stats': self.get_performance_stats(),
            'check_interval': self.check_interval
        }
    
    def shutdown(self):
        """Clean shutdown of configuration manager"""
        try:
            self.logger.info("Shutting down Configuration Manager...")
            
            # Stop monitoring
            self.is_monitoring = False
            
            # Wait for monitoring thread to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            # Save any pending changes
            if self.pending_changes:
                self.logger.info("Saving pending configuration changes...")
                for (scope, config_name), config_data in self.pending_changes.items():
                    try:
                        self.set_configuration(scope, config_name, config_data, save_to_file=True)
                    except Exception as e:
                        self.logger.error(f"Error saving pending change {scope}.{config_name}: {e}")
            
            # Log final statistics
            final_stats = self.get_performance_stats()
            self.logger.info(f"Final configuration statistics: {final_stats}")
            
            self.status = "SHUTDOWN"
            self.logger.info("Configuration Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Agent test
if __name__ == "__main__":
    # Test the configuration manager
    print("Testing AGENT_12: Configuration Manager")
    print("=" * 40)
    
    # Create configuration manager
    config_manager = ConfigurationManager("test_config")
    result = config_manager.initialize()
    print(f"Initialization: {result}")
    
    if result['status'] == 'initialized':
        # Test setting configuration
        print("\nTesting configuration management...")
        
        test_config = {
            'connection_timeout': 30,
            'retry_attempts': 3,
            'auto_reconnect': True,
            'heartbeat_interval': 60
        }
        
        set_result = config_manager.set_configuration('agents', 'mt5_connector', test_config)
        print(f"Set configuration: {set_result}")
        
        # Test getting configuration
        retrieved_config = config_manager.get_configuration('agents', 'mt5_connector')
        print(f"Retrieved configuration: {retrieved_config}")
        
        # Test configuration watcher
        def config_changed(name, data):
            print(f"Configuration changed: {name}")
        
        watcher_result = config_manager.add_configuration_watcher('mt5_connector', config_changed)
        print(f"Watcher added: {watcher_result}")
        
        # Test export/import
        export_result = config_manager.export_configuration('agents', 'mt5_connector')
        print(f"Export result: {export_result['status']}")
        
        # Test performance stats
        stats = config_manager.get_performance_stats()
        print(f"Performance stats: {stats}")
        
        # Test status
        status = config_manager.get_status()
        print(f"\nStatus: {status}")
        
        # Shutdown
        print("\nShutting down...")
        config_manager.shutdown()
        
    print("Configuration Manager test completed")