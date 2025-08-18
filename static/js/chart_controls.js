// Chart controls and interactions
class ChartControls {
    constructor() {
        this.currentSymbol = 'EURUSD';
        this.currentTimeframe = 'H1';
        this.chart = null;
        this.indicators = {
            sma: false,
            ema: false,
            rsi: false,
            macd: false
        };
    }

    init() {
        this.setupSymbolSelector();
        this.setupTimeframeButtons();
        this.setupIndicatorToggles();
        this.loadChart();
    }

    setupSymbolSelector() {
        const selector = document.getElementById('symbol-selector');
        if (selector) {
            selector.addEventListener('change', (e) => {
                this.currentSymbol = e.target.value;
                this.loadChart();
            });
        }
    }

    setupTimeframeButtons() {
        const buttons = document.querySelectorAll('.timeframe-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Remove active class from all buttons
                buttons.forEach(b => b.classList.remove('active'));
                
                // Add active class to clicked button
                e.target.classList.add('active');
                
                this.currentTimeframe = e.target.dataset.timeframe;
                this.loadChart();
            });
        });
    }

    setupIndicatorToggles() {
        const toggles = document.querySelectorAll('.indicator-toggle');
        toggles.forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                const indicator = e.target.dataset.indicator;
                this.indicators[indicator] = e.target.checked;
                this.updateIndicators();
            });
        });
    }

    loadChart() {
        const chartContainer = document.getElementById('chart-container');
        if (!chartContainer) return;

        // Show loading indicator
        this.showLoading(chartContainer);

        // Fetch chart data
        fetch(`/api/chart/${this.currentSymbol}/${this.currentTimeframe}`)
            .then(response => response.json())
            .then(data => {
                this.renderChart(data);
            })
            .catch(error => {
                console.error('Error loading chart:', error);
                this.showError(chartContainer, 'Failed to load chart data');
            });
    }

    renderChart(data) {
        const chartContainer = document.getElementById('chart-container');
        
        // Clear loading indicator
        chartContainer.innerHTML = '';

        // Create Plotly chart
        const traces = [{
            x: data.timestamps,
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.close,
            type: 'candlestick',
            name: this.currentSymbol,
            increasing: { line: { color: '#00ff00' } },
            decreasing: { line: { color: '#ff0000' } }
        }];

        const layout = {
            title: `${this.currentSymbol} - ${this.currentTimeframe}`,
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: { color: '#ffffff' },
            xaxis: {
                gridcolor: '#404040',
                type: 'date'
            },
            yaxis: {
                gridcolor: '#404040',
                title: 'Price'
            }
        };

        const config = {
            responsive: true,
            displayModeBar: true
        };

        Plotly.newPlot(chartContainer, traces, layout, config);

        // Update price display
        this.updatePriceDisplay(data);
    }

    updateIndicators() {
        // Update chart with selected indicators
        // This would fetch indicator data and add to chart
        console.log('Updating indicators:', this.indicators);
    }

    updatePriceDisplay(data) {
        const latest = data.close[data.close.length - 1];
        const previous = data.close[data.close.length - 2];
        const change = latest - previous;
        const changePercent = (change / previous) * 100;

        const priceValue = document.getElementById('price-value');
        const priceChange = document.getElementById('price-change');

        if (priceValue) {
            priceValue.textContent = latest.toFixed(5);
        }

        if (priceChange) {
            priceChange.textContent = `${change > 0 ? '+' : ''}${change.toFixed(5)} (${changePercent.toFixed(2)}%)`;
            priceChange.className = `price-change ${change > 0 ? 'price-up' : 'price-down'}`;
        }
    }

    showLoading(container) {
        container.innerHTML = '<div class="loading">Loading chart data...</div>';
    }

    showError(container, message) {
        container.innerHTML = `<div class="error">Error: ${message}</div>`;
    }
}

// Initialize chart controls
document.addEventListener('DOMContentLoaded', function() {
    const chartControls = new ChartControls();
    chartControls.init();
});