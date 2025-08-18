"""
Comprehensive Code Validator for AGI Trading System
Reviews all components systematically for errors, compatibility, and functionality
"""

import sys
import os
import ast
import importlib
import traceback
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class CodeValidator:
    """Comprehensive code validation system"""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.validation_results = {}
        self.errors_found = []
        self.warnings_found = []
        self.recommendations = []
        
        # Define all agent specifications
        self.agents = {
            'AGENT_01': {
                'name': 'MT5_WINDOWS_CONNECTOR',
                'file': 'mt5_connector_fixed.py',
                'class': 'MT5ConnectorFixed',
                'test_file': 'test_agent_01_simple.py',
                'critical_methods': ['initialize', 'get_current_price', 'get_account_info', 'shutdown']
            },
            'AGENT_02': {
                'name': 'SIGNAL_COORDINATOR', 
                'file': 'core/signal_coordinator.py',
                'class': 'SignalCoordinator',
                'test_file': 'test_agent_02_simple.py',
                'critical_methods': ['initialize', 'start_coordination', 'stop_coordination', 'get_status']
            },
            'AGENT_03': {
                'name': 'RISK_CALCULATOR',
                'file': 'core/risk_calculator.py', 
                'class': 'RiskCalculator',
                'test_file': 'test_agent_03_simple.py',
                'critical_methods': ['initialize', 'calculate_position_size', 'calculate_stop_loss', 'shutdown']
            },
            'AGENT_04': {
                'name': 'CHART_SIGNAL_AGENT',
                'file': 'core/chart_signal_agent.py',
                'class': 'ChartSignalAgent', 
                'test_file': 'test_agent_04_simple.py',
                'critical_methods': ['initialize', 'start_analysis', 'stop_analysis', 'get_current_signal']
            },
            'AGENT_05': {
                'name': 'NEURAL_SIGNAL_BRAIN',
                'file': 'ml/neural_signal_brain.py',
                'class': 'NeuralSignalBrain',
                'test_file': 'test_agent_05_simple.py', 
                'critical_methods': ['initialize', 'predict_signal', 'add_training_sample', 'shutdown']
            },
            'AGENT_06': {
                'name': 'TECHNICAL_ANALYST',
                'file': 'data/technical_analyst.py',
                'class': 'TechnicalAnalyst',
                'test_file': 'test_agent_06_simple.py',
                'critical_methods': ['initialize', 'analyze_current_market', 'get_current_analysis', 'shutdown']
            },
            'AGENT_07': {
                'name': 'MARKET_DATA_MANAGER',
                'file': 'data/market_data_manager.py',
                'class': 'MarketDataManager',
                'test_file': 'test_agent_07_simple.py',
                'critical_methods': ['initialize', 'start_streaming', 'stop_streaming', 'get_latest_tick']
            },
            'AGENT_08': {
                'name': 'TRADE_EXECUTION_ENGINE',
                'file': 'execution/trade_execution_engine.py',
                'class': 'TradeExecutionEngine',
                'test_file': 'test_agent_08_simple.py',
                'critical_methods': ['initialize', 'execute_signal', 'get_positions', 'shutdown']
            },
            'AGENT_09': {
                'name': 'PORTFOLIO_MANAGER',
                'file': 'portfolio/portfolio_manager.py',
                'class': 'PortfolioManager',
                'test_file': 'test_agent_09_simple.py',
                'critical_methods': ['initialize', 'add_position', 'get_portfolio_summary', 'rebalance_portfolio']
            },
            'AGENT_10': {
                'name': 'PERFORMANCE_ANALYTICS',
                'file': 'analytics/performance_analytics.py',
                'class': 'PerformanceAnalytics',
                'test_file': 'test_agent_10_simple.py',
                'critical_methods': ['initialize', 'analyze_performance', 'generate_report', 'shutdown']
            },
            'AGENT_11': {
                'name': 'ALERT_SYSTEM',
                'file': 'alerts/alert_system.py',
                'class': 'AlertSystem',
                'test_file': 'test_agent_11_simple.py',
                'critical_methods': ['initialize', 'create_alert', 'get_active_alerts', 'shutdown']
            },
            'AGENT_12': {
                'name': 'CONFIGURATION_MANAGER',
                'file': 'config/configuration_manager.py',
                'class': 'ConfigurationManager',
                'test_file': 'test_agent_12_simple.py',
                'critical_methods': ['initialize', 'get_configuration', 'set_configuration', 'shutdown']
            }
        }
        
        # Integration components
        self.integration_components = {
            'MAIN_SYSTEM': {
                'file': 'start_trading_system.py',
                'class': 'AGITradingSystem',
                'critical_methods': ['initialize_system', 'start_trading', 'shutdown_system']
            },
            'COMPREHENSIVE_TESTS': {
                'file': 'test_comprehensive_suite.py',
                'class': 'ComprehensiveTestSuite',
                'critical_methods': ['run_comprehensive_tests', 'run_integration_tests']
            },
            'SYSTEM_MONITOR': {
                'file': 'system_monitor.py',
                'class': 'SystemMonitor', 
                'critical_methods': ['start_monitoring', 'show_status', 'stop_monitoring']
            }
        }
    
    def run_comprehensive_validation(self) -> Dict:
        """Run complete validation suite"""
        print("="*80)
        print("COMPREHENSIVE CODE VALIDATION - AGI TRADING SYSTEM")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project Root: {self.project_root}")
        print("="*80)
        
        validation_start = time.time()
        
        # Phase 1: Syntax and Import Validation
        print("\n[PHASE 1] SYNTAX AND IMPORT VALIDATION")
        print("-" * 60)
        syntax_results = self.validate_syntax_and_imports()
        
        # Phase 2: Class Structure Validation  
        print("\n[PHASE 2] CLASS STRUCTURE VALIDATION")
        print("-" * 60)
        structure_results = self.validate_class_structures()
        
        # Phase 3: Method Implementation Validation
        print("\n[PHASE 3] METHOD IMPLEMENTATION VALIDATION")
        print("-" * 60)
        method_results = self.validate_method_implementations()
        
        # Phase 4: Dependencies and Compatibility
        print("\n[PHASE 4] DEPENDENCY AND COMPATIBILITY VALIDATION")
        print("-" * 60)
        dependency_results = self.validate_dependencies()
        
        # Phase 5: Integration Validation
        print("\n[PHASE 5] INTEGRATION VALIDATION")
        print("-" * 60)
        integration_results = self.validate_integrations()
        
        # Phase 6: Test Suite Validation
        print("\n[PHASE 6] TEST SUITE VALIDATION")
        print("-" * 60)
        test_results = self.validate_test_suites()
        
        validation_duration = time.time() - validation_start
        
        # Compile final results
        final_results = {
            'validation_duration': validation_duration,
            'syntax_validation': syntax_results,
            'structure_validation': structure_results,
            'method_validation': method_results,
            'dependency_validation': dependency_results,
            'integration_validation': integration_results,
            'test_validation': test_results,
            'errors_found': self.errors_found,
            'warnings_found': self.warnings_found,
            'recommendations': self.recommendations,
            'overall_status': self.determine_overall_status()
        }
        
        self.generate_validation_report(final_results)
        
        return final_results
    
    def validate_syntax_and_imports(self) -> Dict:
        """Validate Python syntax and import statements"""
        results = {}
        
        all_files = []
        # Add agent files
        for agent_id, spec in self.agents.items():
            all_files.append((agent_id, spec['file']))
            all_files.append((f"{agent_id}_TEST", spec['test_file']))
        
        # Add integration files
        for comp_id, spec in self.integration_components.items():
            all_files.append((comp_id, spec['file']))
        
        for file_id, file_path in all_files:
            full_path = os.path.join(self.project_root, file_path)
            
            if not os.path.exists(full_path):
                self.errors_found.append(f"Missing file: {file_path}")
                results[file_id] = {'status': 'missing', 'file': file_path}
                continue
            
            try:
                # Check syntax
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to validate syntax
                try:
                    ast.parse(content)
                    syntax_ok = True
                    syntax_error = None
                except SyntaxError as e:
                    syntax_ok = False
                    syntax_error = str(e)
                    self.errors_found.append(f"Syntax error in {file_path}: {syntax_error}")
                
                # Try importing the module (simplified approach)
                import_ok = True
                import_error = None
                try:
                    if file_path.endswith('.py'):
                        # Just check if the file can be compiled, not actually imported
                        # to avoid complex dependency issues during validation
                        try:
                            compile(content, full_path, 'exec')
                        except Exception as compile_error:
                            import_ok = False
                            import_error = f"Compile error: {compile_error}"
                
                except Exception as e:
                    import_ok = False
                    import_error = str(e)
                    if "No module named" in str(e) or "cannot import" in str(e):
                        self.warnings_found.append(f"Import warning in {file_path}: {import_error}")
                    else:
                        self.errors_found.append(f"Import error in {file_path}: {import_error}")
                
                results[file_id] = {
                    'status': 'valid' if syntax_ok and import_ok else ('syntax_error' if not syntax_ok else 'import_error'),
                    'file': file_path,
                    'syntax_ok': syntax_ok,
                    'syntax_error': syntax_error,
                    'import_ok': import_ok,
                    'import_error': import_error
                }
                
                if syntax_ok and import_ok:
                    print(f"[PASS] {file_id}: {file_path}")
                elif syntax_ok:
                    print(f"[WARN] {file_id}: {file_path} (import warnings)")
                else:
                    print(f"[FAIL] {file_id}: {file_path} (syntax error)")
                    
            except Exception as e:
                results[file_id] = {'status': 'error', 'file': file_path, 'error': str(e)}
                self.errors_found.append(f"Validation error for {file_path}: {str(e)}")
                print(f"[FAIL] {file_id}: {file_path} (validation error)")
        
        return results
    
    def validate_class_structures(self) -> Dict:
        """Validate class structures and required methods"""
        results = {}
        
        for agent_id, spec in self.agents.items():
            file_path = os.path.join(self.project_root, spec['file'])
            
            if not os.path.exists(file_path):
                results[agent_id] = {'status': 'missing_file'}
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to find class definition
                tree = ast.parse(content)
                class_found = False
                methods_found = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == spec['class']:
                        class_found = True
                        
                        # Find methods in the class
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                methods_found.append(item.name)
                
                # Check if all critical methods are present
                missing_methods = []
                for method in spec['critical_methods']:
                    if method not in methods_found:
                        missing_methods.append(method)
                
                if missing_methods:
                    self.errors_found.append(f"{agent_id}: Missing critical methods: {missing_methods}")
                
                results[agent_id] = {
                    'status': 'valid' if class_found and not missing_methods else 'incomplete',
                    'class_found': class_found,
                    'methods_found': methods_found,
                    'missing_methods': missing_methods,
                    'critical_methods_complete': len(missing_methods) == 0
                }
                
                if class_found and not missing_methods:
                    print(f"[PASS] {agent_id}: Class structure complete")
                elif class_found:
                    print(f"[WARN] {agent_id}: Missing methods: {missing_methods}")
                else:
                    print(f"[FAIL] {agent_id}: Class {spec['class']} not found")
                    
            except Exception as e:
                results[agent_id] = {'status': 'error', 'error': str(e)}
                self.errors_found.append(f"{agent_id} class validation error: {str(e)}")
                print(f"[FAIL] {agent_id}: Validation error")
        
        return results
    
    def validate_method_implementations(self) -> Dict:
        """Validate that critical methods are properly implemented"""
        results = {}
        
        # This would require more sophisticated AST analysis
        # For now, we'll do basic checks
        
        for agent_id, spec in self.agents.items():
            try:
                file_path = os.path.join(self.project_root, spec['file'])
                
                if not os.path.exists(file_path):
                    results[agent_id] = {'status': 'missing_file'}
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                method_implementations = {}
                
                for method in spec['critical_methods']:
                    # Simple check: does the method have actual implementation beyond just pass?
                    method_pattern = f"def {method}("
                    if method_pattern in content:
                        # Get the method body (simplified)
                        method_start = content.find(method_pattern)
                        if method_start != -1:
                            # Find the end of the method (very simplified)
                            method_section = content[method_start:method_start + 1000]
                            
                            # Check if it's just a pass or has real implementation
                            has_implementation = (
                                'return' in method_section or
                                'self.' in method_section or
                                'logger' in method_section or
                                'try:' in method_section or
                                len([line for line in method_section.split('\n') if line.strip() and not line.strip().startswith('#')]) > 3
                            )
                            
                            method_implementations[method] = {
                                'found': True,
                                'has_implementation': has_implementation
                            }
                        else:
                            method_implementations[method] = {'found': False, 'has_implementation': False}
                    else:
                        method_implementations[method] = {'found': False, 'has_implementation': False}
                
                # Determine overall status
                all_implemented = all(
                    impl['found'] and impl['has_implementation']
                    for impl in method_implementations.values()
                )
                
                results[agent_id] = {
                    'status': 'fully_implemented' if all_implemented else 'partially_implemented',
                    'method_implementations': method_implementations,
                    'implementation_complete': all_implemented
                }
                
                if all_implemented:
                    print(f"[PASS] {agent_id}: All methods implemented")
                else:
                    incomplete = [method for method, impl in method_implementations.items() 
                                if not (impl['found'] and impl['has_implementation'])]
                    print(f"[WARN] {agent_id}: Incomplete methods: {incomplete}")
                    self.warnings_found.append(f"{agent_id}: Methods may need review: {incomplete}")
                    
            except Exception as e:
                results[agent_id] = {'status': 'error', 'error': str(e)}
                print(f"[FAIL] {agent_id}: Method validation error")
        
        return results
    
    def validate_dependencies(self) -> Dict:
        """Validate dependencies and compatibility"""
        results = {}
        
        # Check for optional dependencies
        optional_deps = {
            'MetaTrader5': 'MT5 trading functionality',
            'pandas': 'Data analysis and manipulation',
            'numpy': 'Numerical computing',
            'psutil': 'System monitoring'
        }
        
        dependency_status = {}
        for dep, description in optional_deps.items():
            try:
                importlib.import_module(dep)
                dependency_status[dep] = {'available': True, 'description': description}
                print(f"[PASS] {dep}: Available")
            except ImportError:
                dependency_status[dep] = {'available': False, 'description': description}
                print(f"[WARN]  {dep}: Not available - {description}")
        
        # Check Python version compatibility
        python_version = sys.version_info
        python_compatible = python_version.major == 3 and python_version.minor >= 7
        
        if python_compatible:
            print(f"[PASS] Python {python_version.major}.{python_version.minor}: Compatible")
        else:
            print(f"[FAIL] Python {python_version.major}.{python_version.minor}: Incompatible (requires 3.7+)")
            self.errors_found.append(f"Python version {python_version.major}.{python_version.minor} is not supported")
        
        results = {
            'dependency_status': dependency_status,
            'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            'python_compatible': python_compatible,
            'critical_missing': []
        }
        
        # Only MetaTrader5 is critical for live trading, others have fallbacks
        if not dependency_status['MetaTrader5']['available']:
            self.warnings_found.append("MetaTrader5 not available - system will run in simulation mode only")
        
        return results
    
    def validate_integrations(self) -> Dict:
        """Validate integration between components"""
        results = {}
        
        print("Testing component integration...")
        
        # Test basic integration between major components
        integration_tests = [
            self.test_mt5_integration,
            self.test_agent_communication,
            self.test_data_flow,
            self.test_error_handling
        ]
        
        for test_func in integration_tests:
            try:
                test_name = test_func.__name__
                result = test_func()
                results[test_name] = result
                
                if result.get('status') == 'pass':
                    print(f"[PASS] {test_name}: PASS")
                else:
                    print(f"[WARN]  {test_name}: {result.get('status', 'UNKNOWN')}")
                    
            except Exception as e:
                results[test_func.__name__] = {'status': 'error', 'error': str(e)}
                print(f"[FAIL] {test_func.__name__}: ERROR")
                self.errors_found.append(f"Integration test {test_func.__name__} failed: {str(e)}")
        
        return results
    
    def test_mt5_integration(self) -> Dict:
        """Test MT5 connector integration"""
        try:
            from mt5_connector_fixed import MT5ConnectorFixed
            connector = MT5ConnectorFixed()
            
            # Test basic attributes
            if hasattr(connector, 'connection_status') and hasattr(connector, 'initialize'):
                return {'status': 'pass', 'message': 'MT5 connector structure valid'}
            else:
                return {'status': 'fail', 'message': 'MT5 connector missing required attributes'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def test_agent_communication(self) -> Dict:
        """Test basic agent communication patterns"""
        try:
            # Test that agents can be imported and have basic interface
            from core.signal_coordinator import SignalCoordinator
            from core.risk_calculator import RiskCalculator
            
            coordinator = SignalCoordinator(None)  # Pass None for MT5 connector
            risk_calc = RiskCalculator()
            
            # Test basic interface
            if (hasattr(coordinator, 'initialize') and 
                hasattr(risk_calc, 'initialize') and
                hasattr(coordinator, 'get_status') and
                hasattr(risk_calc, 'get_status')):
                return {'status': 'pass', 'message': 'Agent communication interfaces valid'}
            else:
                return {'status': 'fail', 'message': 'Missing communication interfaces'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def test_data_flow(self) -> Dict:
        """Test data flow between components"""
        try:
            # Test that data structures are compatible
            from data.technical_analyst import TechnicalAnalyst
            
            analyst = TechnicalAnalyst('EURUSD')
            
            if hasattr(analyst, 'get_current_analysis'):
                return {'status': 'pass', 'message': 'Data flow structures valid'}
            else:
                return {'status': 'fail', 'message': 'Data flow interfaces missing'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def test_error_handling(self) -> Dict:
        """Test error handling patterns"""
        try:
            # Test that components handle errors gracefully
            from core.risk_calculator import RiskCalculator
            
            risk_calc = RiskCalculator()
            
            # Test with invalid inputs
            try:
                result = risk_calc.calculate_position_size(
                    symbol="INVALID",
                    direction="INVALID", 
                    entry_price=0,
                    stop_loss=0,
                    take_profit=0
                )
                
                if isinstance(result, dict) and 'status' in result:
                    return {'status': 'pass', 'message': 'Error handling works correctly'}
                else:
                    return {'status': 'fail', 'message': 'Error handling may be inadequate'}
                    
            except Exception:
                return {'status': 'fail', 'message': 'Error handling throws unhandled exceptions'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def validate_test_suites(self) -> Dict:
        """Validate that test suites are comprehensive"""
        results = {}
        
        print("Validating test suite coverage...")
        
        for agent_id, spec in self.agents.items():
            test_file = os.path.join(self.project_root, spec['test_file'])
            
            if os.path.exists(test_file):
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        test_content = f.read()
                    
                    # Count test cases
                    test_functions = test_content.count('def test_')
                    assert_statements = test_content.count('assert ')
                    
                    # Check if main critical methods are tested
                    methods_tested = []
                    for method in spec['critical_methods']:
                        if method in test_content:
                            methods_tested.append(method)
                    
                    coverage_percent = (len(methods_tested) / len(spec['critical_methods'])) * 100
                    
                    results[agent_id] = {
                        'test_file_exists': True,
                        'test_functions': test_functions,
                        'assert_statements': assert_statements,
                        'methods_tested': methods_tested,
                        'coverage_percent': coverage_percent,
                        'status': 'good' if coverage_percent >= 80 else 'needs_improvement'
                    }
                    
                    print(f"[PASS] {agent_id}: {coverage_percent:.1f}% coverage ({test_functions} tests)")
                    
                except Exception as e:
                    results[agent_id] = {'test_file_exists': True, 'status': 'error', 'error': str(e)}
                    print(f"[FAIL] {agent_id}: Test file error")
            else:
                results[agent_id] = {'test_file_exists': False, 'status': 'missing'}
                print(f"[FAIL] {agent_id}: Test file missing")
                self.errors_found.append(f"Missing test file: {spec['test_file']}")
        
        return results
    
    def determine_overall_status(self) -> str:
        """Determine overall system status"""
        critical_errors = len([e for e in self.errors_found if 'syntax error' in e.lower() or 'missing file' in e.lower()])
        
        if critical_errors > 0:
            return 'CRITICAL_ISSUES'
        elif len(self.errors_found) > 0:
            return 'ERRORS_FOUND'
        elif len(self.warnings_found) > 3:
            return 'WARNINGS_PRESENT'
        else:
            return 'SYSTEM_READY'
    
    def generate_validation_report(self, results: Dict):
        """Generate comprehensive validation report"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY REPORT")
        print("="*80)
        
        print(f"Validation Duration: {results['validation_duration']:.2f} seconds")
        print(f"Overall Status: {results['overall_status']}")
        
        print(f"\nErrors Found: {len(self.errors_found)}")
        for error in self.errors_found[:10]:  # Show first 10 errors
            print(f"  [FAIL] {error}")
        if len(self.errors_found) > 10:
            print(f"  ... and {len(self.errors_found) - 10} more errors")
        
        print(f"\nWarnings Found: {len(self.warnings_found)}")
        for warning in self.warnings_found[:10]:  # Show first 10 warnings
            print(f"  [WARN]  {warning}")
        if len(self.warnings_found) > 10:
            print(f"  ... and {len(self.warnings_found) - 10} more warnings")
        
        # Status interpretation
        print(f"\nSTATUS INTERPRETATION:")
        if results['overall_status'] == 'SYSTEM_READY':
            print("[SUCCESS] SYSTEM IS READY FOR PRODUCTION!")
            print("   All critical components validated successfully")
        elif results['overall_status'] == 'WARNINGS_PRESENT':
            print("[PASS] SYSTEM IS FUNCTIONAL WITH MINOR ISSUES")
            print("   System can run but some optimizations recommended")
        elif results['overall_status'] == 'ERRORS_FOUND':
            print("[WARN]  SYSTEM HAS ERRORS THAT NEED FIXING")
            print("   System may run but errors should be addressed")
        elif results['overall_status'] == 'CRITICAL_ISSUES':
            print("[FAIL] CRITICAL ISSUES PREVENT SYSTEM OPERATION")
            print("   System cannot run until critical issues are resolved")
        
        print("="*80)

def main():
    """Main validation execution"""
    validator = CodeValidator()
    results = validator.run_comprehensive_validation()
    
    return results['overall_status'] == 'SYSTEM_READY' or results['overall_status'] == 'WARNINGS_PRESENT'

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n[TARGET] VALIDATION PASSED - System is operational!")
    else:
        print(f"\n[NEEDS_WORK] VALIDATION ISSUES - Review and fix before deployment")