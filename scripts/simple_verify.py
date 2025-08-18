"""Simple verification script without Unicode characters"""

import os
import sys

def verify_project_structure():
    """Check if all directories exist"""
    required_dirs = [
        'core', 'ml', 'data', 'ui', 'utils', 'validation',
        'config', 'static', 'templates', 'scripts', 'logs',
        'database', 'models', 'docs'
    ]
    
    print("Verifying project structure...")
    missing_dirs = []
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"OK: {directory}/")
        else:
            print(f"MISSING: {directory}/")
            missing_dirs.append(directory)
    return len(missing_dirs) == 0

def verify_agent_templates():
    """Check if all 12 agent templates exist"""
    agent_files = [
        'core/mt5_windows_connector.py',
        'core/signal_coordinator.py', 
        'core/risk_calculator.py',
        'core/chart_signal_agent.py',
        'ml/neural_signal_brain.py',
        'data/technical_analyst.py',
        'data/news_sentiment_reader.py',
        'data/learning_optimizer.py',
        'ui/professional_dashboard.py',
        'data/data_stream_manager.py',
        'utils/system_monitor.py',
        'validation/signal_validator.py'
    ]
    
    print("\nVerifying agent templates...")
    missing_agents = []
    for agent_file in agent_files:
        if os.path.exists(agent_file):
            print(f"OK: {agent_file}")
        else:
            print(f"MISSING: {agent_file}")
            missing_agents.append(agent_file)
    return len(missing_agents) == 0

def verify_config_files():
    """Check if all configuration files exist"""
    config_files = [
        'config/mt5_windows_config.yaml',
        'config/chart_selection.yaml',
        'config/risk_parameters.yaml',
        'config/indicator_settings.yaml',
        'config/news_sources.yaml',
        'config/agent_registry.yaml',
        'config/ui_settings.yaml'
    ]
    
    print("\nVerifying configuration files...")
    missing_configs = []
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"OK: {config_file}")
        else:
            print(f"MISSING: {config_file}")
            missing_configs.append(config_file)
    return len(missing_configs) == 0

def test_agent_imports():
    """Test that agents can be imported"""
    print("\nTesting agent imports...")
    try:
        sys.path.append('.')
        from core.mt5_windows_connector import MT5WindowsConnector
        from core.signal_coordinator import SignalCoordinator
        from ml.neural_signal_brain import NeuralSignalBrain
        print("OK: Core agents importable")
        return True
    except Exception as e:
        print(f"ERROR: Agent import failed: {e}")
        return False

if __name__ == "__main__":
    print("AGI Trading System - Setup Verification")
    print("=" * 50)
    
    structure_ok = verify_project_structure()
    agents_ok = verify_agent_templates()
    configs_ok = verify_config_files()
    imports_ok = test_agent_imports()
    
    print("\n" + "=" * 50)
    print("Verification Summary:")
    print(f"Project Structure: {'PASS' if structure_ok else 'FAIL'}")
    print(f"Agent Templates: {'PASS' if agents_ok else 'FAIL'}")
    print(f"Configuration Files: {'PASS' if configs_ok else 'FAIL'}")
    print(f"Agent Imports: {'PASS' if imports_ok else 'FAIL'}")
    
    if all([structure_ok, agents_ok, configs_ok, imports_ok]):
        print("\nVERIFICATION SUCCESSFUL!")
        print("Ready for AGI development phase!")
        success_rate = 100
    else:
        failed_checks = sum([not structure_ok, not agents_ok, not configs_ok, not imports_ok])
        success_rate = ((4 - failed_checks) / 4) * 100
        print(f"\nVERIFICATION PARTIAL: {success_rate:.1f}% success rate")
    
    print(f"\nFinal Status: {success_rate:.1f}% Setup Complete")