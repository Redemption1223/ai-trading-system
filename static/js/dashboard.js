// Main dashboard functionality
class Dashboard {
    constructor() {
        this.updateInterval = 1000; // 1 second
        this.metrics = {
            totalSignals: 0,
            accuracy: 0,
            profit: 0,
            activeAgents: 0
        };
    }

    init() {
        this.setupAutoRefresh();
        this.loadInitialData();
        this.setupUIHandlers();
    }

    setupAutoRefresh() {
        setInterval(() => {
            this.updateMetrics();
            this.updateAgentStatus();
        }, this.updateInterval);
    }

    loadInitialData() {
        this.updateMetrics();
        this.updateAgentStatus();
        this.loadRecentSignals();
    }

    updateMetrics() {
        fetch('/api/metrics')
            .then(response => response.json())
            .then(data => {
                this.metrics = data;
                this.displayMetrics();
            })
            .catch(error => {
                console.error('Error updating metrics:', error);
            });
    }

    displayMetrics() {
        // Update metric cards
        const elements = {
            'signal-count': this.metrics.totalSignals,
            'accuracy-metric': `${(this.metrics.accuracy * 100).toFixed(1)}%`,
            'profit-metric': `$${this.metrics.profit.toFixed(2)}`,
            'active-agents': this.metrics.activeAgents
        };

        Object.keys(elements).forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = elements[id];
            }
        });

        // Update profit color
        const profitElement = document.getElementById('profit-metric');
        if (profitElement) {
            profitElement.className = this.metrics.profit >= 0 ? 'profit-positive' : 'profit-negative';
        }
    }

    updateAgentStatus() {
        fetch('/api/agents/status')
            .then(response => response.json())
            .then(data => {
                this.displayAgentStatus(data);
            })
            .catch(error => {
                console.error('Error updating agent status:', error);
            });
    }

    displayAgentStatus(agents) {
        const statusContainer = document.getElementById('agent-status-list');
        if (!statusContainer) return;

        statusContainer.innerHTML = '';

        agents.forEach(agent => {
            const agentElement = document.createElement('div');
            agentElement.className = 'agent-status-item';
            agentElement.innerHTML = `
                <div class="agent-info">
                    <span class="status-indicator status-${agent.status}"></span>
                    <span class="agent-name">${agent.name}</span>
                </div>
                <div class="agent-metrics">
                    <small>Signals: ${agent.signalsGenerated || 0}</small>
                    <small>Accuracy: ${((agent.accuracy || 0) * 100).toFixed(1)}%</small>
                </div>
            `;
            statusContainer.appendChild(agentElement);
        });
    }

    loadRecentSignals() {
        fetch('/api/signals/recent')
            .then(response => response.json())
            .then(signals => {
                this.displayRecentSignals(signals);
            })
            .catch(error => {
                console.error('Error loading recent signals:', error);
            });
    }

    displayRecentSignals(signals) {
        const signalContainer = document.getElementById('signal-list');
        if (!signalContainer) return;

        signalContainer.innerHTML = '';

        signals.forEach(signal => {
            const signalElement = document.createElement('div');
            signalElement.className = `signal-item signal-${signal.type}`;
            signalElement.innerHTML = `
                <div class="signal-header">
                    <strong>${signal.symbol}</strong>
                    <span class="signal-time">${this.formatTime(signal.timestamp)}</span>
                </div>
                <div class="signal-details">
                    <div>Type: ${signal.type.toUpperCase()}</div>
                    <div>Confidence: ${(signal.confidence * 100).toFixed(1)}%</div>
                    <div>Agent: ${signal.agent}</div>
                </div>
            `;
            signalContainer.appendChild(signalElement);
        });
    }

    setupUIHandlers() {
        // Handle panel resizing
        this.setupPanelResize();
        
        // Handle theme switching
        this.setupThemeSwitch();
        
        // Handle alert settings
        this.setupAlertSettings();
    }

    setupPanelResize() {
        // Allow panels to be resized
        const panels = document.querySelectorAll('.panel');
        panels.forEach(panel => {
            panel.addEventListener('mousedown', this.startResize);
        });
    }

    setupThemeSwitch() {
        const themeSwitch = document.getElementById('theme-switch');
        if (themeSwitch) {
            themeSwitch.addEventListener('change', (e) => {
                document.body.classList.toggle('light-theme', e.target.checked);
            });
        }
    }

    setupAlertSettings() {
        const alertToggle = document.getElementById('alerts-enabled');
        if (alertToggle) {
            alertToggle.addEventListener('change', (e) => {
                localStorage.setItem('alertsEnabled', e.target.checked);
            });
        }
    }

    formatTime(timestamp) {
        return new Date(timestamp).toLocaleTimeString();
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    const dashboard = new Dashboard();
    dashboard.init();
});