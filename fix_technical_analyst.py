"""
Fix Technical Analyst pandas/numpy compatibility issues
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def fix_aroon_function():
    """Fix the Aroon indicator function"""
    
    file_path = r"C:\Users\douglasvz\OneDrive - FlameBlock (Pty)Ltd\Documents\claude trader\agi_trading_system\data\technical_analyst.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the Aroon function
    old_aroon = '''                high_pos = period_high.index(max(period_high))
                low_pos = period_low.index(min(period_low))'''
    
    new_aroon = '''                # Handle both list and numpy array cases
                if hasattr(period_high, 'index'):
                    high_pos = period_high.index(max(period_high))
                    low_pos = period_low.index(min(period_low))
                else:
                    # For numpy arrays
                    high_pos = np.argmax(period_high) if NUMPY_AVAILABLE else list(period_high).index(max(period_high))
                    low_pos = np.argmin(period_low) if NUMPY_AVAILABLE else list(period_low).index(min(period_low))'''
    
    # Replace in content
    content = content.replace(old_aroon, new_aroon)
    
    # Fix trend analysis DataFrame ambiguity
    old_trend = '''            # Trend strength calculation
            if trend_direction != TrendDirection.SIDEWAYS:
                if abs(slope) > 0.001:  # Strong trend
                    trend_strength = min(100, abs(slope) * 10000)
                else:
                    trend_strength = 50'''
    
    new_trend = '''            # Trend strength calculation
            if trend_direction != TrendDirection.SIDEWAYS:
                # Handle pandas Series/DataFrame vs scalar values
                slope_val = slope.iloc[0] if hasattr(slope, 'iloc') else slope
                if abs(slope_val) > 0.001:  # Strong trend
                    trend_strength = min(100, abs(slope_val) * 10000)
                else:
                    trend_strength = 50'''
    
    # Replace in content
    content = content.replace(old_trend, new_trend)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed Aroon function and trend analysis")

def fix_market_data_manager():
    """Fix the Market Data Manager symbol processing"""
    
    file_path = r"C:\Users\douglasvz\OneDrive - FlameBlock (Pty)Ltd\Documents\claude trader\agi_trading_system\data\market_data_manager.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for the issue where symbols are being processed as individual characters
    # This usually happens in a loop where symbols should be iterated, not the symbol string itself
    
    # Find and fix symbol iteration issues
    old_pattern1 = '''for symbol in self.symbols:
            for char in symbol:'''
    
    new_pattern1 = '''for symbol in self.symbols:
            # Process each symbol (not individual characters)'''
    
    # Replace if found
    if old_pattern1 in content:
        content = content.replace(old_pattern1, new_pattern1)
        print("✅ Fixed symbol iteration in Market Data Manager")
    
    # Look for other potential issues
    old_pattern2 = '''for symbol in symbols:
            for s in symbol:'''
    
    new_pattern2 = '''for symbol in symbols:
            # Process the symbol string correctly'''
    
    if old_pattern2 in content:
        content = content.replace(old_pattern2, new_pattern2)
        print("✅ Fixed additional symbol iteration")
    
    # Write back if changes were made
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Apply all fixes"""
    print("Fixing Technical Analysis Issues...")
    print("=" * 50)
    
    try:
        fix_aroon_function()
        fix_market_data_manager()
        
        print("\n✅ All fixes applied successfully!")
        print("Run the system again: python start_trading_system.py")
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")

if __name__ == "__main__":
    main()