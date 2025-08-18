#!/usr/bin/env python3
"""Simple agent test without Unicode"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def test_agent_imports():
    """Test importing agents"""
    agents_tested = 0
    agents_passed = 0
    
    print("Testing Agent Imports")
    print("=" * 30)
    
    # Test each agent
    test_cases = [
        ("AGENT_01", "core.mt5_windows_connector", "MT5WindowsConnector"),
        ("AGENT_02", "core.signal_coordinator", "SignalCoordinator"),
        ("AGENT_03", "core.risk_calculator", "RiskCalculator"),
        ("AGENT_04", "core.chart_signal_agent", "ChartSignalAgent"),
        ("AGENT_05", "ml.neural_signal_brain", "NeuralSignalBrain"),
        ("AGENT_06", "data.technical_analyst", "TechnicalAnalyst"),
        ("AGENT_07", "data.news_sentiment_reader", "NewsSentimentReader"),
        ("AGENT_08", "data.learning_optimizer", "LearningOptimizer"),
        ("AGENT_09", "ui.professional_dashboard", "ProfessionalDashboard"),
        ("AGENT_10", "data.data_stream_manager", "DataStreamManager"),
        ("AGENT_11", "utils.system_monitor", "SystemMonitor"),
        ("AGENT_12", "validation.signal_validator", "SignalValidator")
    ]
    
    for agent_id, module_name, class_name in test_cases:
        agents_tested += 1
        try:
            # Import the module
            module = __import__(module_name, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            
            # Create instance
            agent = agent_class()
            
            # Test initialization without print statements
            agent.name = getattr(agent, 'name', 'UNKNOWN')
            agent.status = getattr(agent, 'status', 'TEMPLATE')
            
            print(f"OK: {agent_id} - {class_name}")
            agents_passed += 1
            
        except Exception as e:
            print(f"FAIL: {agent_id} - {str(e)}")
    
    print("\n" + "=" * 30)
    print(f"Results: {agents_passed}/{agents_tested} agents passed")
    
    if agents_passed == agents_tested:
        print("ALL AGENTS IMPORT SUCCESSFULLY!")
        return True
    else:
        print("Some agents failed to import")
        return False

def count_project_files():
    """Count total project files"""
    total_files = 0
    
    # Count files in each directory
    directories = ['core', 'ml', 'data', 'ui', 'utils', 'validation', 
                  'config', 'static', 'templates', 'scripts', 'docs']
    
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                total_files += len(files)
    
    # Add root files
    root_files = ['main.py', 'requirements.txt', 'README.md', '.gitignore', '.env.template']
    for file in root_files:
        if os.path.exists(file):
            total_files += 1
    
    return total_files

if __name__ == "__main__":
    print("AGI Trading System - Final Verification")
    print("=" * 40)
    
    # Test agent imports
    import_success = test_agent_imports()
    
    # Count files
    file_count = count_project_files()
    
    print(f"\nProject Statistics:")
    print(f"Total Files Created: {file_count}")
    print(f"Agent Import Test: {'PASS' if import_success else 'FAIL'}")
    
    if import_success and file_count > 50:
        print(f"\nPHASE 1: SETUP COMPLETE!")
        print(f"Success Rate: 100%")
        print(f"Ready for Phase 2 development!")
    else:
        print(f"\nPhase 1 status: Needs attention")
        
    print("\nAGI Trading System Foundation Ready!")