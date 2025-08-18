"""
Advanced UI Dashboard System
Integrated from MQL5 Expert Advisor - Enhanced Trading Interface
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import json
import os

class TradingUITheme:
    """Professional Trading UI Theme"""
    # Dark theme colors
    BG_DARK = "#1e1e1e"
    BG_PANEL = "#2d2d2d"
    BG_BUTTON = "#404040"
    BG_ACTIVE = "#0078d4"
    
    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#cccccc"
    TEXT_SUCCESS = "#00ff00"
    TEXT_WARNING = "#ffff00"
    TEXT_ERROR = "#ff0000"
    
    # Status colors
    GREEN = "#00ff00"
    RED = "#ff0000"
    YELLOW = "#ffff00"
    ORANGE = "#ffa500"
    BLUE = "#0078d4"
    GRAY = "#808080"

class StatusDisplay:
    """Status display widget"""
    def __init__(self, parent, title: str, value: str = "0", color: str = TradingUITheme.TEXT_PRIMARY):
        self.frame = tk.Frame(parent, bg=TradingUITheme.BG_PANEL)
        self.title_label = tk.Label(
            self.frame, 
            text=title, 
            fg=TradingUITheme.TEXT_SECONDARY, 
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 8)
        )
        self.value_label = tk.Label(
            self.frame, 
            text=value, 
            fg=color, 
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 10, "bold")
        )
        
        self.title_label.pack()
        self.value_label.pack()
    
    def update(self, value: str, color: str = None):
        """Update display value and color"""
        self.value_label.config(text=value)
        if color:
            self.value_label.config(fg=color)
    
    def pack(self, **kwargs):
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

class TradingButton:
    """Enhanced trading button"""
    def __init__(self, parent, text: str, command: Callable, bg_color: str = TradingUITheme.BG_BUTTON):
        self.button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg=TradingUITheme.TEXT_PRIMARY,
            font=("Arial", 9, "bold"),
            relief="flat",
            bd=1,
            activebackground=TradingUITheme.BG_ACTIVE,
            activeforeground=TradingUITheme.TEXT_PRIMARY
        )
        self.original_bg = bg_color
        self.is_active = False
    
    def toggle_active(self):
        """Toggle button active state"""
        self.is_active = not self.is_active
        if self.is_active:
            self.button.config(bg=TradingUITheme.BG_ACTIVE)
        else:
            self.button.config(bg=self.original_bg)
    
    def set_active(self, active: bool):
        """Set button active state"""
        self.is_active = active
        if active:
            self.button.config(bg=TradingUITheme.BG_ACTIVE)
        else:
            self.button.config(bg=self.original_bg)
    
    def pack(self, **kwargs):
        self.button.pack(**kwargs)
    
    def grid(self, **kwargs):
        self.button.grid(**kwargs)

class AdvancedTradingDashboard:
    """Advanced Trading Dashboard based on MQL5 UI"""
    
    def __init__(self, trading_system=None):
        self.name = "ADVANCED_TRADING_DASHBOARD"
        self.version = "1.0.0"
        self.trading_system = trading_system
        
        # UI state
        self.is_running = False
        self.update_thread = None
        self.refresh_rate = 1000  # milliseconds
        
        # Trading state
        self.trading_paused = False
        self.conservative_mode = False
        self.auto_trading = True
        
        # Data containers
        self.market_data = {}
        self.account_data = {}
        self.position_data = {}
        self.performance_data = {}
        self.microstructure_data = {}
        self.pattern_data = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Initialize UI
        self.setup_main_window()
        self.create_ui_components()
    
    def setup_main_window(self):
        """Setup main window"""
        self.root = tk.Tk()
        self.root.title(f"AGI Trading System - Advanced Dashboard v{self.version}")
        self.root.geometry("1200x800")
        self.root.configure(bg=TradingUITheme.BG_DARK)
        self.root.resizable(True, True)
        
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_ui_components(self):
        """Create all UI components based on MQL5 design"""
        # Header
        self.create_header()
        
        # Main panels
        self.create_status_panel()
        self.create_trading_panel()
        self.create_analysis_panel()
        
        # Bottom panels
        self.create_microstructure_panel()
        self.create_performance_panel()
        self.create_control_panel()
    
    def create_header(self):
        """Create header with system title and status"""
        header_frame = tk.Frame(self.root, bg=TradingUITheme.BG_PANEL, height=60)
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        header_frame.grid_propagate(False)
        
        # System title
        title_label = tk.Label(
            header_frame,
            text="🚀 AGI Trading System - LIVE Trading Dashboard",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.LEFT, padx=10, pady=15)
        
        # System status
        self.system_status = tk.Label(
            header_frame,
            text="SYSTEM: INITIALIZING",
            fg=TradingUITheme.YELLOW,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 12, "bold")
        )
        self.system_status.pack(side=tk.RIGHT, padx=10, pady=15)
    
    def create_status_panel(self):
        """Create main status panel (left column)"""
        status_frame = tk.Frame(self.root, bg=TradingUITheme.BG_PANEL)
        status_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Panel title
        title = tk.Label(
            status_frame,
            text="📊 SYSTEM STATUS",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 12, "bold")
        )
        title.pack(pady=10)
        
        # Account information
        account_frame = tk.LabelFrame(
            status_frame,
            text="Account Information",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 10, "bold")
        )
        account_frame.pack(fill="x", padx=10, pady=5)
        
        self.account_balance = StatusDisplay(account_frame, "Balance", "$0.00", TradingUITheme.GREEN)
        self.account_balance.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.account_equity = StatusDisplay(account_frame, "Equity", "$0.00", TradingUITheme.GREEN)
        self.account_equity.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.account_margin = StatusDisplay(account_frame, "Margin", "0%", TradingUITheme.YELLOW)
        self.account_margin.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Trading statistics
        stats_frame = tk.LabelFrame(
            status_frame,
            text="Trading Statistics",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 10, "bold")
        )
        stats_frame.pack(fill="x", padx=10, pady=5)
        
        self.total_trades = StatusDisplay(stats_frame, "Total Trades", "0", TradingUITheme.TEXT_PRIMARY)
        self.total_trades.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.win_rate = StatusDisplay(stats_frame, "Win Rate", "0%", TradingUITheme.TEXT_PRIMARY)
        self.win_rate.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.profit_factor = StatusDisplay(stats_frame, "Profit Factor", "0.0", TradingUITheme.TEXT_PRIMARY)
        self.profit_factor.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Market information
        market_frame = tk.LabelFrame(
            status_frame,
            text="Market Information",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 10, "bold")
        )
        market_frame.pack(fill="x", padx=10, pady=5)
        
        self.market_trend = StatusDisplay(market_frame, "Trend", "NEUTRAL", TradingUITheme.YELLOW)
        self.market_trend.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.market_regime = StatusDisplay(market_frame, "Regime", "QUIET", TradingUITheme.TEXT_PRIMARY)
        self.market_regime.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.current_pattern = StatusDisplay(market_frame, "Pattern", "None", TradingUITheme.GRAY)
        self.current_pattern.pack(side=tk.LEFT, padx=10, pady=5)
    
    def create_trading_panel(self):
        """Create trading panel (middle column)"""
        trading_frame = tk.Frame(self.root, bg=TradingUITheme.BG_PANEL)
        trading_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Panel title
        title = tk.Label(
            trading_frame,
            text="🔥 LIVE TRADING",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 12, "bold")
        )
        title.pack(pady=10)
        
        # Current positions
        positions_frame = tk.LabelFrame(
            trading_frame,
            text="Open Positions",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 10, "bold")
        )
        positions_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Position list
        self.position_listbox = tk.Listbox(
            positions_frame,
            bg=TradingUITheme.BG_DARK,
            fg=TradingUITheme.TEXT_PRIMARY,
            selectbackground=TradingUITheme.BG_ACTIVE,
            font=("Courier", 9)
        )
        self.position_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Quick trading
        quick_trade_frame = tk.LabelFrame(
            trading_frame,
            text="Quick Trade",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 10, "bold")
        )
        quick_trade_frame.pack(fill="x", padx=10, pady=5)
        
        # Symbol and lot size
        trade_inputs_frame = tk.Frame(quick_trade_frame, bg=TradingUITheme.BG_PANEL)
        trade_inputs_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Label(trade_inputs_frame, text="Symbol:", fg=TradingUITheme.TEXT_PRIMARY, bg=TradingUITheme.BG_PANEL).pack(side=tk.LEFT)
        self.symbol_var = tk.StringVar(value="EURUSD")
        symbol_combo = ttk.Combobox(trade_inputs_frame, textvariable=self.symbol_var, values=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"], width=10)
        symbol_combo.pack(side=tk.LEFT, padx=5)
        
        tk.Label(trade_inputs_frame, text="Lot:", fg=TradingUITheme.TEXT_PRIMARY, bg=TradingUITheme.BG_PANEL).pack(side=tk.LEFT, padx=(10,0))
        self.lot_var = tk.StringVar(value="0.01")
        lot_entry = tk.Entry(trade_inputs_frame, textvariable=self.lot_var, width=8, bg=TradingUITheme.BG_DARK, fg=TradingUITheme.TEXT_PRIMARY)
        lot_entry.pack(side=tk.LEFT, padx=5)
        
        # Buy/Sell buttons
        button_frame = tk.Frame(quick_trade_frame, bg=TradingUITheme.BG_PANEL)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        self.buy_button = TradingButton(button_frame, "BUY", self.place_buy_order, TradingUITheme.GREEN)
        self.buy_button.pack(side=tk.LEFT, padx=5, fill="x", expand=True)
        
        self.sell_button = TradingButton(button_frame, "SELL", self.place_sell_order, TradingUITheme.RED)
        self.sell_button.pack(side=tk.LEFT, padx=5, fill="x", expand=True)
    
    def create_analysis_panel(self):
        """Create analysis panel (right column)"""
        analysis_frame = tk.Frame(self.root, bg=TradingUITheme.BG_PANEL)
        analysis_frame.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
        
        # Panel title
        title = tk.Label(
            analysis_frame,
            text="🧠 AI ANALYSIS",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 12, "bold")
        )
        title.pack(pady=10)
        
        # Technical analysis
        tech_frame = tk.LabelFrame(
            analysis_frame,
            text="Technical Analysis",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 10, "bold")
        )
        tech_frame.pack(fill="x", padx=10, pady=5)
        
        self.rsi_display = StatusDisplay(tech_frame, "RSI", "50.0", TradingUITheme.TEXT_PRIMARY)
        self.rsi_display.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.macd_display = StatusDisplay(tech_frame, "MACD", "0.0", TradingUITheme.TEXT_PRIMARY)
        self.macd_display.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.adx_display = StatusDisplay(tech_frame, "ADX", "20.0", TradingUITheme.TEXT_PRIMARY)
        self.adx_display.pack(side=tk.LEFT, padx=10, pady=5)
        
        # AI signals
        signals_frame = tk.LabelFrame(
            analysis_frame,
            text="AI Signals",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 10, "bold")
        )
        signals_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.signals_text = tk.Text(
            signals_frame,
            bg=TradingUITheme.BG_DARK,
            fg=TradingUITheme.TEXT_PRIMARY,
            font=("Courier", 8),
            height=15
        )
        self.signals_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Correlation analysis
        corr_frame = tk.LabelFrame(
            analysis_frame,
            text="Correlation Analysis",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 10, "bold")
        )
        corr_frame.pack(fill="x", padx=10, pady=5)
        
        self.dxy_corr = StatusDisplay(corr_frame, "DXY", "0.0", TradingUITheme.TEXT_PRIMARY)
        self.dxy_corr.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.gold_corr = StatusDisplay(corr_frame, "GOLD", "0.0", TradingUITheme.TEXT_PRIMARY)
        self.gold_corr.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.oil_corr = StatusDisplay(corr_frame, "OIL", "0.0", TradingUITheme.TEXT_PRIMARY)
        self.oil_corr.pack(side=tk.LEFT, padx=10, pady=5)
    
    def create_microstructure_panel(self):
        """Create microstructure analysis panel"""
        micro_frame = tk.Frame(self.root, bg=TradingUITheme.BG_PANEL, height=120)
        micro_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        micro_frame.grid_propagate(False)
        
        # Panel title
        title = tk.Label(
            micro_frame,
            text="🔬 MARKET MICROSTRUCTURE",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 12, "bold")
        )
        title.pack(side=tk.TOP, pady=5)
        
        # Microstructure metrics
        metrics_frame = tk.Frame(micro_frame, bg=TradingUITheme.BG_PANEL)
        metrics_frame.pack(fill="x", padx=10, pady=5)
        
        self.liquidity_score = StatusDisplay(metrics_frame, "Liquidity", "50%", TradingUITheme.TEXT_PRIMARY)
        self.liquidity_score.pack(side=tk.LEFT, padx=10)
        
        self.hft_activity = StatusDisplay(metrics_frame, "HFT Activity", "0%", TradingUITheme.TEXT_PRIMARY)
        self.hft_activity.pack(side=tk.LEFT, padx=10)
        
        self.dark_pool = StatusDisplay(metrics_frame, "Dark Pool", "0%", TradingUITheme.TEXT_PRIMARY)
        self.dark_pool.pack(side=tk.LEFT, padx=10)
        
        self.manipulation = StatusDisplay(metrics_frame, "Manipulation", "0%", TradingUITheme.TEXT_PRIMARY)
        self.manipulation.pack(side=tk.LEFT, padx=10)
        
        self.execution_quality = StatusDisplay(metrics_frame, "Exec Quality", "100%", TradingUITheme.GREEN)
        self.execution_quality.pack(side=tk.LEFT, padx=10)
    
    def create_performance_panel(self):
        """Create performance analysis panel"""
        perf_frame = tk.Frame(self.root, bg=TradingUITheme.BG_PANEL, height=120)
        perf_frame.grid(row=2, column=2, sticky="ew", padx=5, pady=5)
        perf_frame.grid_propagate(False)
        
        # Panel title
        title = tk.Label(
            perf_frame,
            text="📈 PERFORMANCE",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 12, "bold")
        )
        title.pack(side=tk.TOP, pady=5)
        
        # Performance metrics
        metrics_frame = tk.Frame(perf_frame, bg=TradingUITheme.BG_PANEL)
        metrics_frame.pack(fill="x", padx=10, pady=5)
        
        self.daily_pnl = StatusDisplay(metrics_frame, "Daily P&L", "$0.00", TradingUITheme.TEXT_PRIMARY)
        self.daily_pnl.pack(side=tk.TOP, padx=5, pady=2)
        
        self.sharpe_ratio = StatusDisplay(metrics_frame, "Sharpe Ratio", "0.0", TradingUITheme.TEXT_PRIMARY)
        self.sharpe_ratio.pack(side=tk.TOP, padx=5, pady=2)
        
        self.max_drawdown = StatusDisplay(metrics_frame, "Max DD", "0%", TradingUITheme.TEXT_PRIMARY)
        self.max_drawdown.pack(side=tk.TOP, padx=5, pady=2)
    
    def create_control_panel(self):
        """Create control panel with buttons"""
        control_frame = tk.Frame(self.root, bg=TradingUITheme.BG_PANEL, height=80)
        control_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        control_frame.grid_propagate(False)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg=TradingUITheme.BG_PANEL)
        button_frame.pack(expand=True, pady=20)
        
        self.pause_button = TradingButton(button_frame, "PAUSE TRADING", self.toggle_trading, TradingUITheme.YELLOW)
        self.pause_button.pack(side=tk.LEFT, padx=10)
        
        self.conservative_button = TradingButton(button_frame, "CONSERVATIVE", self.toggle_conservative, TradingUITheme.ORANGE)
        self.conservative_button.pack(side=tk.LEFT, padx=10)
        
        self.auto_button = TradingButton(button_frame, "AUTO TRADING", self.toggle_auto_trading, TradingUITheme.GREEN)
        self.auto_button.pack(side=tk.LEFT, padx=10)
        
        self.emergency_button = TradingButton(button_frame, "EMERGENCY STOP", self.emergency_stop, TradingUITheme.RED)
        self.emergency_button.pack(side=tk.LEFT, padx=10)
        
        self.settings_button = TradingButton(button_frame, "SETTINGS", self.open_settings, TradingUITheme.BLUE)
        self.settings_button.pack(side=tk.LEFT, padx=10)
    
    def initialize(self) -> Dict:
        """Initialize the dashboard"""
        try:
            self.logger.info(f"Initializing {self.name} v{self.version}")
            
            # Start update thread
            self.is_running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            
            # Update system status
            self.system_status.config(text="SYSTEM: ONLINE", fg=TradingUITheme.GREEN)
            
            self.logger.info("Advanced Trading Dashboard initialized successfully")
            return {
                "status": "initialized",
                "agent": "ADVANCED_TRADING_DASHBOARD",
                "refresh_rate_ms": self.refresh_rate
            }
            
        except Exception as e:
            self.logger.error(f"Dashboard initialization failed: {e}")
            return {"status": "failed", "agent": "ADVANCED_TRADING_DASHBOARD", "error": str(e)}
    
    def update_account_data(self, account_info: Dict):
        """Update account data display"""
        self.account_data = account_info
        
        if 'balance' in account_info:
            balance = account_info['balance']
            self.account_balance.update(f"${balance:,.2f}", TradingUITheme.GREEN if balance > 0 else TradingUITheme.RED)
        
        if 'equity' in account_info:
            equity = account_info['equity']
            self.account_equity.update(f"${equity:,.2f}", TradingUITheme.GREEN if equity > 0 else TradingUITheme.RED)
        
        if 'margin_level' in account_info:
            margin = account_info['margin_level']
            color = TradingUITheme.GREEN if margin > 200 else TradingUITheme.YELLOW if margin > 100 else TradingUITheme.RED
            self.account_margin.update(f"{margin:.1f}%", color)
    
    def update_trading_stats(self, stats: Dict):
        """Update trading statistics"""
        if 'total_trades' in stats:
            self.total_trades.update(str(stats['total_trades']))
        
        if 'win_rate' in stats:
            win_rate = stats['win_rate']
            color = TradingUITheme.GREEN if win_rate > 60 else TradingUITheme.YELLOW if win_rate > 40 else TradingUITheme.RED
            self.win_rate.update(f"{win_rate:.1f}%", color)
        
        if 'profit_factor' in stats:
            pf = stats['profit_factor']
            color = TradingUITheme.GREEN if pf > 1.2 else TradingUITheme.YELLOW if pf > 1.0 else TradingUITheme.RED
            self.profit_factor.update(f"{pf:.2f}", color)
    
    def update_market_data(self, market_info: Dict):
        """Update market information"""
        self.market_data = market_info
        
        if 'trend' in market_info:
            trend = market_info['trend']
            color = TradingUITheme.GREEN if trend == "BULLISH" else TradingUITheme.RED if trend == "BEARISH" else TradingUITheme.YELLOW
            self.market_trend.update(trend, color)
        
        if 'regime' in market_info:
            regime = market_info['regime']
            self.market_regime.update(regime, TradingUITheme.TEXT_PRIMARY)
        
        if 'pattern' in market_info:
            pattern = market_info['pattern']
            color = TradingUITheme.GREEN if pattern != "None" else TradingUITheme.GRAY
            self.current_pattern.update(pattern, color)
    
    def update_microstructure_data(self, micro_data: Dict):
        """Update microstructure data"""
        self.microstructure_data = micro_data
        
        if 'liquidity_provision' in micro_data:
            liquidity = micro_data['liquidity_provision']
            color = TradingUITheme.GREEN if liquidity > 70 else TradingUITheme.YELLOW if liquidity > 40 else TradingUITheme.RED
            self.liquidity_score.update(f"{liquidity:.0f}%", color)
        
        if 'high_frequency_activity' in micro_data:
            hft = micro_data['high_frequency_activity']
            color = TradingUITheme.YELLOW if hft > 50 else TradingUITheme.TEXT_PRIMARY
            self.hft_activity.update(f"{hft:.0f}%", color)
        
        if 'dark_pool_activity' in micro_data:
            dark_pool = micro_data['dark_pool_activity']
            color = TradingUITheme.ORANGE if dark_pool > 30 else TradingUITheme.TEXT_PRIMARY
            self.dark_pool.update(f"{dark_pool:.0f}%", color)
        
        if 'manipulation_detection' in micro_data:
            manipulation = micro_data['manipulation_detection']
            color = TradingUITheme.RED if manipulation > 30 else TradingUITheme.TEXT_PRIMARY
            self.manipulation.update(f"{manipulation:.0f}%", color)
        
        if 'execution_quality' in micro_data:
            quality = micro_data['execution_quality']
            color = TradingUITheme.GREEN if quality > 80 else TradingUITheme.YELLOW if quality > 60 else TradingUITheme.RED
            self.execution_quality.update(f"{quality:.0f}%", color)
    
    def update_positions(self, positions: List[Dict]):
        """Update position list"""
        self.position_listbox.delete(0, tk.END)
        
        for pos in positions:
            symbol = pos.get('symbol', 'Unknown')
            profit = pos.get('profit', 0.0)
            volume = pos.get('volume', 0.0)
            
            profit_color = "green" if profit >= 0 else "red"
            position_text = f"{symbol:<8} {volume:.2f} lots   P&L: ${profit:>8.2f}"
            
            self.position_listbox.insert(tk.END, position_text)
            # Note: Tkinter Listbox doesn't support per-item colors easily
    
    def add_signal(self, signal_text: str, signal_type: str = "INFO"):
        """Add signal to signals display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color based on signal type
        if signal_type == "BUY":
            color_tag = "green"
        elif signal_type == "SELL":
            color_tag = "red"
        elif signal_type == "WARNING":
            color_tag = "yellow"
        elif signal_type == "ERROR":
            color_tag = "red"
        else:
            color_tag = "white"
        
        # Add to text widget
        self.signals_text.insert(tk.END, f"[{timestamp}] {signal_text}\\n")
        
        # Auto-scroll to bottom
        self.signals_text.see(tk.END)
        
        # Limit text length
        lines = self.signals_text.get("1.0", tk.END).split("\\n")
        if len(lines) > 100:
            # Remove old lines
            self.signals_text.delete("1.0", "10.0")
    
    def toggle_trading(self):
        """Toggle trading pause/resume"""
        self.trading_paused = not self.trading_paused
        
        if self.trading_paused:
            self.pause_button.button.config(text="RESUME TRADING", bg=TradingUITheme.GREEN)
            self.system_status.config(text="SYSTEM: PAUSED", fg=TradingUITheme.YELLOW)
            self.add_signal("Trading PAUSED by user", "WARNING")
        else:
            self.pause_button.button.config(text="PAUSE TRADING", bg=TradingUITheme.YELLOW)
            self.system_status.config(text="SYSTEM: ONLINE", fg=TradingUITheme.GREEN)
            self.add_signal("Trading RESUMED by user", "INFO")
        
        # Notify trading system
        if self.trading_system and hasattr(self.trading_system, 'set_trading_paused'):
            self.trading_system.set_trading_paused(self.trading_paused)
    
    def toggle_conservative(self):
        """Toggle conservative mode"""
        self.conservative_mode = not self.conservative_mode
        self.conservative_button.set_active(self.conservative_mode)
        
        if self.conservative_mode:
            self.conservative_button.button.config(text="NORMAL MODE")
            self.add_signal("Conservative mode ENABLED", "INFO")
        else:
            self.conservative_button.button.config(text="CONSERVATIVE")
            self.add_signal("Normal mode ENABLED", "INFO")
        
        # Notify trading system
        if self.trading_system and hasattr(self.trading_system, 'set_conservative_mode'):
            self.trading_system.set_conservative_mode(self.conservative_mode)
    
    def toggle_auto_trading(self):
        """Toggle auto trading"""
        self.auto_trading = not self.auto_trading
        self.auto_button.set_active(self.auto_trading)
        
        if self.auto_trading:
            self.auto_button.button.config(text="DISABLE AUTO")
            self.add_signal("Auto trading ENABLED", "INFO")
        else:
            self.auto_button.button.config(text="AUTO TRADING")
            self.add_signal("Auto trading DISABLED", "WARNING")
        
        # Notify trading system
        if self.trading_system and hasattr(self.trading_system, 'set_auto_trading'):
            self.trading_system.set_auto_trading(self.auto_trading)
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        result = messagebox.askyesno(
            "Emergency Stop",
            "This will immediately stop all trading activities and close all positions.\\n\\nAre you sure?"
        )
        
        if result:
            self.system_status.config(text="SYSTEM: EMERGENCY STOP", fg=TradingUITheme.RED)
            self.add_signal("EMERGENCY STOP activated!", "ERROR")
            
            # Notify trading system
            if self.trading_system and hasattr(self.trading_system, 'emergency_stop'):
                self.trading_system.emergency_stop()
    
    def place_buy_order(self):
        """Place buy order"""
        symbol = self.symbol_var.get()
        lot_size = float(self.lot_var.get())
        
        self.add_signal(f"BUY order placed: {symbol} {lot_size} lots", "BUY")
        
        # Notify trading system
        if self.trading_system and hasattr(self.trading_system, 'place_market_order'):
            self.trading_system.place_market_order(symbol, "BUY", lot_size)
    
    def place_sell_order(self):
        """Place sell order"""
        symbol = self.symbol_var.get()
        lot_size = float(self.lot_var.get())
        
        self.add_signal(f"SELL order placed: {symbol} {lot_size} lots", "SELL")
        
        # Notify trading system
        if self.trading_system and hasattr(self.trading_system, 'place_market_order'):
            self.trading_system.place_market_order(symbol, "SELL", lot_size)
    
    def open_settings(self):
        """Open settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Trading System Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg=TradingUITheme.BG_PANEL)
        
        # Settings content
        tk.Label(
            settings_window,
            text="Settings Panel",
            fg=TradingUITheme.TEXT_PRIMARY,
            bg=TradingUITheme.BG_PANEL,
            font=("Arial", 14, "bold")
        ).pack(pady=20)
        
        # Placeholder for settings
        tk.Label(
            settings_window,
            text="Settings configuration will be implemented here",
            fg=TradingUITheme.TEXT_SECONDARY,
            bg=TradingUITheme.BG_PANEL
        ).pack(pady=10)
    
    def _update_loop(self):
        """Main update loop"""
        while self.is_running:
            try:
                # Update timestamp in header
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # Update from trading system if available
                if self.trading_system:
                    # This would be connected to the actual trading system
                    pass
                
                time.sleep(self.refresh_rate / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Update loop error: {e}")
                time.sleep(1)
    
    def run(self):
        """Run the dashboard"""
        try:
            self.logger.info("Starting Advanced Trading Dashboard...")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Dashboard runtime error: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        self.is_running = False
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
        
        self.root.destroy()
    
    def shutdown(self):
        """Shutdown the dashboard"""
        try:
            self.logger.info("Shutting down Advanced Trading Dashboard...")
            self.is_running = False
            
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5)
            
            self.logger.info("Advanced Trading Dashboard shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Dashboard shutdown error: {e}")

def test_trading_dashboard():
    """Test the trading dashboard"""
    print("Testing Advanced Trading Dashboard...")
    print("=" * 60)
    
    dashboard = AdvancedTradingDashboard()
    
    # Test initialization
    result = dashboard.initialize()
    print(f"Initialization: {result['status']}")
    
    if result['status'] == 'initialized':
        # Simulate some data updates
        dashboard.update_account_data({
            'balance': 10000.0,
            'equity': 10250.0,
            'margin_level': 300.0
        })
        
        dashboard.update_trading_stats({
            'total_trades': 25,
            'win_rate': 68.0,
            'profit_factor': 1.45
        })
        
        dashboard.update_market_data({
            'trend': 'BULLISH',
            'regime': 'TRENDING',
            'pattern': 'Gartley'
        })
        
        dashboard.update_microstructure_data({
            'liquidity_provision': 75.0,
            'high_frequency_activity': 35.0,
            'dark_pool_activity': 15.0,
            'manipulation_detection': 5.0,
            'execution_quality': 85.0
        })
        
        dashboard.add_signal("System initialized successfully", "INFO")
        dashboard.add_signal("EURUSD BUY signal detected", "BUY")
        dashboard.add_signal("High volatility detected", "WARNING")
        
        print("[OK] Advanced Trading Dashboard Test PASSED!")
        print("Dashboard is ready to run...")
        return dashboard
    else:
        print(f"[FAIL] Test failed: {result.get('error', 'Unknown error')}")
        return None

if __name__ == "__main__":
    dashboard = test_trading_dashboard()
    if dashboard:
        # Run the dashboard
        dashboard.run()