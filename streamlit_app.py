import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, Brain, Shield, DollarSign, Activity, Settings, AlertTriangle, CheckCircle, Clock, BarChart3, Zap, Globe, Play, Pause, Target } from 'lucide-react';

const AdvancedTradingDashboard = () => {
  const [marketData, setMarketData] = useState({});
  const [predictions, setPredictions] = useState({});
  const [tradingSignals, setTradingSignals] = useState([]);
  const [patternAnalysis, setPatternAnalysis] = useState({});
  const [accountInfo, setAccountInfo] = useState({});
  const [systemStatus, setSystemStatus] = useState({});
  const [autoTradingEnabled, setAutoTradingEnabled] = useState(false);
  const [selectedPair, setSelectedPair] = useState('EURUSD');
  const [selectedTimeframe, setSelectedTimeframe] = useState('ALL');

  const MAJOR_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD'];
  const MINOR_PAIRS = ['NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP', 'AUDCAD', 'AUDCHF'];
  const EXOTIC_PAIRS = ['AUDJPY', 'AUDNZD', 'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD'];

  const API_URL = "https://ai-trading-system-production.up.railway.app";

  const fetchData = async (endpoint) => {
    try {
      const response = await fetch(`${API_URL}/${endpoint}`);
      return response.ok ? await response.json() : null;
    } catch (error) {
      console.error(`Error fetching ${endpoint}:`, error);
      return null;
    }
  };

  useEffect(() => {
    const fetchAllData = async () => {
      const [market, pred, signals, patterns, account, status] = await Promise.all([
        fetchData('market-data'),
        fetchData('ai-predictions'),
        fetchData('trading-signals'),
        fetchData('pattern-analysis'),
        fetchData('account-info'),
        fetchData('system-status')
      ]);

      if (market) setMarketData(market);
      if (pred) setPredictions(pred);
      if (signals) setTradingSignals(signals);
      if (patterns) setPatternAnalysis(patterns);
      if (account) setAccountInfo(account);
      if (status) {
        setSystemStatus(status);
        setAutoTradingEnabled(status.auto_trading_enabled || false);
      }
    };

    fetchAllData();
    const interval = setInterval(fetchAllData, 5000);
    return () => clearInterval(interval);
  }, []);

  const toggleAutoTrading = async () => {
    try {
      const response = await fetch(`${API_URL}/toggle-auto-trading`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !autoTradingEnabled })
      });
      
      if (response.ok) {
        const result = await response.json();
        setAutoTradingEnabled(result.auto_trading_enabled);
      }
    } catch (error) {
      console.error('Error toggling auto-trading:', error);
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value || 0);
  };

  const formatPrice = (value, symbol) => {
    const decimals = symbol?.includes('JPY') ? 3 : 5;
    return Number(value || 0).toFixed(decimals);
  };

  const getSignalColor = (action) => {
    switch(action?.toLowerCase()) {
      case 'buy': return '#10b981';
      case 'sell': return '#ef4444';
      default: return '#f59e0b';
    }
  };

  const getPatternStrength = (strength) => {
    if (strength > 0.8) return { label: 'Very Strong', color: '#10b981' };
    if (strength > 0.6) return { label: 'Strong', color: '#3b82f6' };
    if (strength > 0.4) return { label: 'Moderate', color: '#f59e0b' };
    return { label: 'Weak', color: '#ef4444' };
  };

  const PairChart = ({ symbol, data, width = "100%", height = 200 }) => {
    if (!data || data.length === 0) {
      return (
        <div className="flex items-center justify-center h-48 bg-gray-800 rounded">
          <span className="text-gray-400">No data for {symbol}</span>
        </div>
      );
    }

    const chartData = data.map(item => ({
      time: new Date(item.timestamp).toLocaleTimeString().slice(0, 5),
      price: item.price,
      bid: item.bid,
      ask: item.ask
    }));

    return (
      <ResponsiveContainer width={width} height={height}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="time" stroke="#9CA3AF" fontSize={10} />
          <YAxis 
            domain={['dataMin - 0.0001', 'dataMax + 0.0001']}
            tickFormatter={(value) => formatPrice(value, symbol)}
            stroke="#9CA3AF"
            fontSize={10}
          />
          <Tooltip 
            labelFormatter={(time) => `Time: ${time}`}
            formatter={(value, name) => [formatPrice(value, symbol), name]}
            contentStyle={{ 
              backgroundColor: '#1F2937', 
              border: '1px solid #374151',
              borderRadius: '8px'
            }}
          />
          <Line 
            type="monotone" 
            dataKey="price" 
            stroke="#8B5CF6" 
            strokeWidth={2}
            dot={false}
            name="Price"
          />
          <Line 
            type="monotone" 
            dataKey="bid" 
            stroke="#10b981" 
            strokeWidth={1}
            dot={false}
            name="Bid"
          />
          <Line 
            type="monotone" 
            dataKey="ask" 
            stroke="#ef4444" 
            strokeWidth={1}
            dot={false}
            name="Ask"
          />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  const TradingSignalCard = ({ signal }) => (
    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 mb-3">
      <div className="flex justify-between items-start mb-2">
        <div className="flex items-center space-x-2">
          <span className="font-bold text-white">{signal.symbol}</span>
          <span 
            className="px-2 py-1 rounded text-xs font-bold text-white"
            style={{ backgroundColor: getSignalColor(signal.action) }}
          >
            {signal.action}
          </span>
          <span className="text-sm text-gray-400">
            {(signal.confidence * 100).toFixed(1)}%
          </span>
        </div>
        <span className="text-xs text-gray-500">
          {new Date(signal.timestamp).toLocaleTimeString()}
        </span>
      </div>
      
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div>
          <span className="text-gray-400">Entry: </span>
          <span className="text-white">{formatPrice(signal.entry_price, signal.symbol)}</span>
        </div>
        <div>
          <span className="text-gray-400">SL: </span>
          <span className="text-red-400">{formatPrice(signal.stop_loss, signal.symbol)}</span>
        </div>
        <div>
          <span className="text-gray-400">TP: </span>
          <span className="text-green-400">{formatPrice(signal.take_profit, signal.symbol)}</span>
        </div>
      </div>
      
      <div className="mt-2 text-xs">
        <span className="text-gray-400">Size: </span>
        <span className="text-white">{signal.position_size} lots</span>
      </div>
      
      <div className="mt-2 text-xs text-gray-300 truncate">
        {signal.reasoning}
      </div>
    </div>
  );

  const PatternCard = ({ symbol, pattern }) => {
    if (!pattern?.pattern) return null;
    
    const strength = getPatternStrength(pattern.pattern?.strength || 0);
    
    return (
      <div className="bg-gray-800 p-3 rounded-lg border border-gray-700">
        <div className="flex justify-between items-center mb-2">
          <span className="font-semibold text-white">{symbol}</span>
          <span 
            className="px-2 py-1 rounded text-xs font-bold"
            style={{ backgroundColor: strength.color, color: 'white' }}
          >
            {strength.label}
          </span>
        </div>
        
        <div className="text-sm text-gray-300 mb-1">
          Pattern: <span className="text-white">{pattern.pattern?.pattern || 'Unknown'}</span>
        </div>
        
        {pattern.pattern?.current_price && (
          <div className="text-sm text-gray-300">
            Price: <span className="text-white">{formatPrice(pattern.pattern.current_price, symbol)}</span>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Brain className="w-8 h-8 text-purple-500" />
            <div>
              <h1 className="text-3xl font-bold">Advanced AI Trading System</h1>
              <p className="text-gray-400">Neural Network | Multi-Currency | Auto-Trading</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Auto-Trading Toggle */}
            <button
              onClick={toggleAutoTrading}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-semibold transition-colors ${
                autoTradingEnabled 
                  ? 'bg-green-600 hover:bg-green-700' 
                  : 'bg-red-600 hover:bg-red-700'
              }`}
            >
              {autoTradingEnabled ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
              <span>{autoTradingEnabled ? 'Auto-Trading ON' : 'Auto-Trading OFF'}</span>
            </button>
            
            {/* Connection Status */}
            <div className={`flex items-center space-x-2 px-4 py-2 rounded-lg ${
              systemStatus.active_pairs > 0 ? 'bg-green-600' : 'bg-red-600'
            }`}>
              <div className="w-3 h-3 rounded-full bg-white opacity-75"></div>
              <span>{systemStatus.active_pairs || 0}/{systemStatus.total_pairs || 20} Pairs Active</span>
            </div>
          </div>
        </div>
      </div>

      {/* Account Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Balance</p>
              <p className="text-2xl font-bold text-white">{formatCurrency(accountInfo.balance)}</p>
            </div>
            <DollarSign className="w-8 h-8 text-green-400" />
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Equity</p>
              <p className="text-2xl font-bold text-blue-400">{formatCurrency(accountInfo.equity)}</p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-400" />
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Profit/Loss</p>
              <p className={`text-2xl font-bold ${(accountInfo.profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(accountInfo.profit)}
              </p>
            </div>
            <Activity className="w-8 h-8 text-purple-400" />
          </div>
        </div>
        
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">AI Predictions</p>
              <p className="text-2xl font-bold text-yellow-400">{systemStatus.total_predictions || 0}</p>
            </div>
            <Brain className="w-8 h-8 text-yellow-400" />
          </div>
        </div>
      </div>

      {/* Main Dashboard */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* Charts Section */}
        <div className="lg:col-span-3 space-y-6">
          
          {/* Pair Selection */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Multi-Currency Charts</h2>
              <select 
                value={selectedPair} 
                onChange={(e) => setSelectedPair(e.target.value)}
                className="bg-gray-700 text-white px-3 py-1 rounded border border-gray-600"
              >
                <optgroup label="Major Pairs">
                  {MAJOR_PAIRS.map(pair => (
                    <option key={pair} value={pair}>{pair}</option>
                  ))}
                </optgroup>
                <optgroup label="Minor Pairs">
                  {MINOR_PAIRS.map(pair => (
                    <option key={pair} value={pair}>{pair}</option>
                  ))}
                </optgroup>
                <optgroup label="Exotic Pairs">
                  {EXOTIC_PAIRS.map(pair => (
                    <option key={pair} value={pair}>{pair}</option>
                  ))}
                </optgroup>
              </select>
            </div>
            
            {/* Main Chart */}
            <div className="h-96">
              <PairChart 
                symbol={selectedPair} 
                data={marketData[selectedPair]} 
                height={350}
              />
            </div>
          </div>

          {/* Major Pairs Overview */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Major Pairs Overview</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {MAJOR_PAIRS.map(pair => (
                <div key={pair} className="bg-gray-700 p-3 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold">{pair}</span>
                    {marketData[pair] && marketData[pair].length > 0 && (
                      <span className="text-sm text-gray-300">
                        {formatPrice(marketData[pair][marketData[pair].length - 1]?.price, pair)}
                      </span>
                    )}
                  </div>
                  <PairChart symbol={pair} data={marketData[pair]} height={120} />
                </div>
              ))}
            </div>
          </div>

          {/* Minor Pairs Overview */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Minor Pairs Overview</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {MINOR_PAIRS.map(pair => (
                <div key={pair} className="bg-gray-700 p-3 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold">{pair}</span>
                    {marketData[pair] && marketData[pair].length > 0 && (
                      <span className="text-sm text-gray-300">
                        {formatPrice(marketData[pair][marketData[pair].length - 1]?.price, pair)}
                      </span>
                    )}
                  </div>
                  <PairChart symbol={pair} data={marketData[pair]} height={120} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Side Panel */}
        <div className="space-y-6">
          
          {/* Trading Signals */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold flex items-center">
                <Target className="w-5 h-5 mr-2" />
                Trading Signals
              </h2>
              <span className="text-sm bg-purple-600 px-2 py-1 rounded">
                {tradingSignals.length}
              </span>
            </div>
            
            <div className="max-h-80 overflow-y-auto">
              {tradingSignals.slice(-10).reverse().map((signal, index) => (
                <TradingSignalCard key={index} signal={signal} />
              ))}
              {tradingSignals.length === 0 && (
                <p className="text-gray-400 text-center py-4">
                  No signals generated yet
                </p>
              )}
            </div>
          </div>

          {/* Pattern Analysis */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <BarChart3 className="w-5 h-5 mr-2" />
              Pattern Analysis
            </h2>
            
            <div className="space-y-3 max-h-60 overflow-y-auto">
              {Object.entries(patternAnalysis).map(([symbol, pattern]) => (
                <PatternCard key={symbol} symbol={symbol} pattern={pattern} />
              ))}
              {Object.keys(patternAnalysis).length === 0 && (
                <p className="text-gray-400 text-center py-4">
                  Analyzing patterns...
                </p>
              )}
            </div>
          </div>

          {/* AI Predictions Summary */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <Brain className="w-5 h-5 mr-2" />
              AI Predictions
            </h2>
            
            <div className="space-y-2">
              {Object.entries(predictions).map(([symbol, preds]) => {
                if (!preds || preds.length === 0) return null;
                const latestPred = preds[preds.length - 1];
                const action = latestPred?.prediction?.action;
                const confidence = latestPred?.prediction?.confidence;
                
                return (
                  <div key={symbol} className="flex justify-between items-center p-2 bg-gray-700 rounded">
                    <span className="text-sm font-medium">{symbol}</span>
                    <div className="flex items-center space-x-2">
                      <span 
                        className="px-2 py-1 rounded text-xs font-bold"
                        style={{ 
                          backgroundColor: getSignalColor(action),
                          color: 'white'
                        }}
                      >
                        {action}
                      </span>
                      <span className="text-xs text-gray-300">
                        {((confidence || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* System Stats */}
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h2 className="text-lg font-semibold mb-4">System Statistics</h2>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-400">Active Pairs</span>
                <span className="text-white">{systemStatus.active_pairs || 0}/20</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-400">Total Predictions</span>
                <span className="text-white">{systemStatus.total_predictions || 0}</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-400">Trading Signals</span>
                <span className="text-white">{systemStatus.trading_signals_count || 0}</span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-400">Auto-Trading</span>
                <span className={autoTradingEnabled ? 'text-green-400' : 'text-red-400'}>
                  {autoTradingEnabled ? 'ENABLED' : 'DISABLED'}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-400">Mode</span>
                <span className="text-yellow-400">
                  {systemStatus.demo_mode ? 'DEMO' : 'LIVE'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-8 p-4 bg-gray-800 rounded-lg border border-gray-700">
        <div className="flex items-center justify-between text-sm text-gray-400">
          <div className="flex items-center space-x-4">
            <span>🧠 Neural AI Processing</span>
            <span>📊 20 Currency Pairs</span>
            <span>⚡ Real-time Predictions</span>
            <span>🛡️ Advanced Risk Management</span>
          </div>
          <div className="text-right">
            <p>Last update: {new Date().toLocaleTimeString()}</p>
            <p className="text-red-400 text-xs">⚠️ Trading involves substantial risk of loss</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedTradingDashboard;
