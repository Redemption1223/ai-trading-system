"""Verify all components are ready for AGI development"""

import os
import importlib
import sys

def verify_project_structure():
    """Check if all directories and files exist"""
    required_dirs = [
        'core', 'ml', 'data', 'ui', 'utils', 'validation',
        'config', 'static', 'templates', 'scripts', 'logs',
        'database', 'models', 'docs'
    ]
    
    print("FOLDERS: Verifying project structure...")
    missing_dirs = []
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"OK: {directory}/")
        else:
            print(f"❌ {directory}/ - MISSING")
            missing_dirs.append(directory)
    return len(missing_dirs) == 0

def verify_dependencies():
    """Check if all packages are installed"""
    critical_packages = [
        'numpy', 'pandas', 'scipy', 'matplotlib',
        'flask', 'plotly', 'requests', 'nltk'
    ]
    
    print("\n📦 Verifying dependencies...")
    missing_packages = []
    for package in critical_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    return len(missing_packages) == 0

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
    
    print("\n🤖 Verifying agent templates...")
    missing_agents = []
    for agent_file in agent_files:
        if os.path.exists(agent_file):
            print(f"✅ {agent_file}")
        else:
            print(f"❌ {agent_file} - MISSING")
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
    
    print("\n⚙️ Verifying configuration files...")
    missing_configs = []
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} - MISSING")
            missing_configs.append(config_file)
    return len(missing_configs) == 0

def test_agent_initialization():
    """Test that all agent templates can be initialized"""
    print("\n🧪 Testing agent initialization...")
    
    # Test a few key agents
    try:
        sys.path.append('.')
        from core.mt5_windows_connector import MT5WindowsConnector
        agent = MT5WindowsConnector()
        result = agent.initialize()
        print(f"✅ MT5WindowsConnector: {result}")
        
        from core.signal_coordinator import SignalCoordinator
        agent2 = SignalCoordinator()
        result2 = agent2.initialize()
        print(f"✅ SignalCoordinator: {result2}")
        
        return True
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("VERIFICATION: AGI Trading System - Setup Verification")
    print("=" * 50)
    
    structure_ok = verify_project_structure()
    dependencies_ok = verify_dependencies()
    agents_ok = verify_agent_templates()
    configs_ok = verify_config_files()
    initialization_ok = test_agent_initialization()
    
    print("\n" + "=" * 50)
    print("📊 Verification Summary:")
    print(f"📁 Project Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"📦 Dependencies: {'✅ PASS' if dependencies_ok else '❌ FAIL'}")
    print(f"🤖 Agent Templates: {'✅ PASS' if agents_ok else '❌ FAIL'}")
    print(f"⚙️ Configuration Files: {'✅ PASS' if configs_ok else '❌ FAIL'}")
    print(f"🧪 Agent Initialization: {'✅ PASS' if initialization_ok else '❌ FAIL'}")
    
    if all([structure_ok, dependencies_ok, agents_ok, configs_ok, initialization_ok]):
        print("\n🎉 VERIFICATION SUCCESSFUL!")
        print("🚀 Ready for AGI development phase!")
        success_rate = 100
    else:
        failed_checks = sum([not structure_ok, not dependencies_ok, not agents_ok, not configs_ok, not initialization_ok])
        success_rate = ((5 - failed_checks) / 5) * 100
        print(f"\n⚠️ VERIFICATION PARTIAL: {success_rate:.1f}% success rate")
        print("🔧 Please fix the issues above before proceeding.")
    
    print(f"\n📋 Final Status: {success_rate:.1f}% Setup Complete")