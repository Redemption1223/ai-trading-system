// Real-time signal updates via WebSocket
class RealtimeSignals {
    constructor() {
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }

    connect() {
        try {
            this.socket = new WebSocket('ws://localhost:5000/signals');
            
            this.socket.onopen = (event) => {
                console.log('✅ Signal WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('online');
            };

            this.socket.onmessage = (event) => {
                const signal = JSON.parse(event.data);
                this.handleSignalUpdate(signal);
            };

            this.socket.onclose = (event) => {
                console.log('❌ Signal WebSocket disconnected');
                this.updateConnectionStatus('offline');
                this.attemptReconnect();
            };

            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('error');
            };

        } catch (error) {
            console.error('Failed to establish WebSocket connection:', error);
        }
    }

    handleSignalUpdate(signal) {
        // Update signal display
        const signalContainer = document.getElementById('signal-list');
        if (signalContainer) {
            const signalElement = this.createSignalElement(signal);
            signalContainer.insertBefore(signalElement, signalContainer.firstChild);
            
            // Keep only last 50 signals
            while (signalContainer.children.length > 50) {
                signalContainer.removeChild(signalContainer.lastChild);
            }
        }

        // Update metrics
        this.updateMetrics(signal);

        // Play alert sound if enabled
        if (signal.importance === 'high') {
            this.playAlert();
        }
    }

    createSignalElement(signal) {
        const element = document.createElement('div');
        element.className = `signal-item signal-${signal.type}`;
        element.innerHTML = `
            <div class="signal-header">
                <strong>${signal.symbol}</strong>
                <span class="signal-time">${new Date(signal.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="signal-details">
                <div>Type: ${signal.type.toUpperCase()}</div>
                <div>Confidence: ${(signal.confidence * 100).toFixed(1)}%</div>
                <div>Source: ${signal.agent}</div>
            </div>
        `;
        return element;
    }

    updateMetrics(signal) {
        // Update signal count
        const countElement = document.getElementById('signal-count');
        if (countElement) {
            const current = parseInt(countElement.textContent) || 0;
            countElement.textContent = current + 1;
        }

        // Update accuracy if provided
        if (signal.accuracy !== undefined) {
            const accuracyElement = document.getElementById('accuracy-metric');
            if (accuracyElement) {
                accuracyElement.textContent = `${(signal.accuracy * 100).toFixed(1)}%`;
            }
        }
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.className = `status-indicator status-${status}`;
        }

        const statusText = document.getElementById('status-text');
        if (statusText) {
            statusText.textContent = status.toUpperCase();
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }

    playAlert() {
        // Play alert sound (placeholder)
        try {
            const audio = new Audio('/static/sounds/alert.wav');
            audio.volume = 0.3;
            audio.play().catch(e => console.log('Could not play alert sound'));
        } catch (e) {
            console.log('Alert sound not available');
        }
    }

    disconnect() {
        if (this.socket) {
            this.socket.close();
        }
    }
}

// Initialize real-time signals when page loads
document.addEventListener('DOMContentLoaded', function() {
    const signals = new RealtimeSignals();
    signals.connect();

    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
        signals.disconnect();
    });
});