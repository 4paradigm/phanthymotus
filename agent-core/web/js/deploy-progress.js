/**
 * deploy-progress.js — WebSocket client for real-time deployment progress
 *
 * Usage:
 *   const monitor = new DeployProgressMonitor(driverId);
 *   monitor.onProgress = (event) => { console.log(event); };
 *   monitor.onError = (event) => { console.error(event); };
 *   monitor.onDone = () => { console.log('Deploy complete'); };
 *   monitor.connect();
 *
 * Event types:
 *   - start: Deployment started
 *   - check: Preflight check result (disk, network, registry)
 *   - progress: Pull/compose/start progress (has percent, speed fields)
 *   - error: Error occurred (has suggestion field)
 *   - done: Deployment complete
 */

class DeployProgressMonitor {
    constructor(driverId) {
        this.driverId = driverId;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 3;

        // Callbacks
        this.onProgress = null;
        this.onError = null;
        this.onDone = null;
        this.onConnected = null;
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const token = localStorage.getItem('access_token') || '';
        const url = `${protocol}//${window.location.host}/ws/deploy/${this.driverId}?token=${token}`;

        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
            console.log(`[DeployProgress] Connected for driver ${this.driverId}`);
            this.reconnectAttempts = 0;
            if (this.onConnected) {
                this.onConnected();
            }
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this._handleMessage(data);
            } catch (e) {
                console.error('[DeployProgress] Failed to parse message:', e);
            }
        };

        this.ws.onerror = (error) => {
            console.error('[DeployProgress] WebSocket error:', error);
        };

        this.ws.onclose = () => {
            console.log(`[DeployProgress] Disconnected for driver ${this.driverId}`);

            // Auto-reconnect on unexpected close
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                console.log(`[DeployProgress] Reconnecting (attempt ${this.reconnectAttempts})...`);
                setTimeout(() => this.connect(), 2000);
            }
        };
    }

    _handleMessage(data) {
        const { type } = data;

        switch (type) {
            case 'connected':
                // Initial connection confirmation
                break;

            case 'ping':
                // Keepalive, ignore
                break;

            case 'start':
                if (this.onProgress) {
                    this.onProgress({ type: 'start', message: data.message, data });
                }
                break;

            case 'check':
                if (this.onProgress) {
                    this.onProgress({
                        type: 'check',
                        checkId: data.check_id,
                        message: data.message,
                        status: data.status, // pass, warning, fail
                        data
                    });
                }
                break;

            case 'progress':
                if (this.onProgress) {
                    this.onProgress({
                        type: 'progress',
                        stage: data.stage, // pull, compose, start
                        message: data.message,
                        percent: data.percent,
                        speed: data.speed,
                        elapsed: data.elapsed,
                        data
                    });
                }
                break;

            case 'error':
                if (this.onError) {
                    this.onError({
                        type: 'error',
                        errorType: data.error_type,
                        message: data.message,
                        suggestion: data.suggestion,
                        data
                    });
                }
                break;

            case 'done':
                if (this.onDone) {
                    this.onDone({ message: data.message, elapsed: data.elapsed, data });
                }
                this.disconnect();
                break;

            default:
                console.warn('[DeployProgress] Unknown message type:', type);
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

/**
 * DeployProgressUI — Simple UI component for showing deployment progress
 *
 * Creates a modal overlay with progress information.
 */
class DeployProgressUI {
    constructor(driverId, driverName) {
        this.driverId = driverId;
        this.driverName = driverName;
        this.monitor = new DeployProgressMonitor(driverId);
        this.container = null;

        this._createUI();
        this._attachCallbacks();
    }

    _createUI() {
        // Create modal overlay
        this.container = document.createElement('div');
        this.container.className = 'deploy-progress-modal';
        this.container.innerHTML = `
            <div class="deploy-progress-content">
                <div class="deploy-progress-header">
                    <h3>部署进度：${this.driverName}</h3>
                    <button class="deploy-progress-close" title="最小化">_</button>
                </div>
                <div class="deploy-progress-body">
                    <div class="deploy-progress-checks"></div>
                    <div class="deploy-progress-main">
                        <div class="deploy-progress-stage"></div>
                        <div class="deploy-progress-bar">
                            <div class="deploy-progress-bar-fill"></div>
                        </div>
                        <div class="deploy-progress-details"></div>
                    </div>
                    <div class="deploy-progress-log"></div>
                </div>
            </div>
        `;

        // Add styles if not already present
        if (!document.getElementById('deploy-progress-styles')) {
            const style = document.createElement('style');
            style.id = 'deploy-progress-styles';
            style.textContent = `
                .deploy-progress-modal {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.5);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 10000;
                }
                .deploy-progress-content {
                    background: #1e1e1e;
                    border-radius: 8px;
                    padding: 20px;
                    min-width: 500px;
                    max-width: 700px;
                    max-height: 80vh;
                    overflow: auto;
                    color: #e0e0e0;
                }
                .deploy-progress-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #333;
                }
                .deploy-progress-header h3 {
                    margin: 0;
                    font-size: 18px;
                }
                .deploy-progress-close {
                    background: none;
                    border: none;
                    color: #999;
                    font-size: 20px;
                    cursor: pointer;
                    padding: 0 10px;
                }
                .deploy-progress-close:hover {
                    color: #fff;
                }
                .deploy-progress-checks {
                    margin-bottom: 15px;
                }
                .deploy-progress-check {
                    padding: 8px;
                    margin: 4px 0;
                    border-radius: 4px;
                    font-size: 14px;
                }
                .deploy-progress-check.pass {
                    background: rgba(76, 175, 80, 0.2);
                    color: #4caf50;
                }
                .deploy-progress-check.warning {
                    background: rgba(255, 152, 0, 0.2);
                    color: #ff9800;
                }
                .deploy-progress-check.fail {
                    background: rgba(244, 67, 54, 0.2);
                    color: #f44336;
                }
                .deploy-progress-stage {
                    font-size: 16px;
                    margin-bottom: 10px;
                    font-weight: 500;
                }
                .deploy-progress-bar {
                    height: 8px;
                    background: #333;
                    border-radius: 4px;
                    overflow: hidden;
                    margin-bottom: 10px;
                }
                .deploy-progress-bar-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #4caf50, #8bc34a);
                    transition: width 0.3s ease;
                    width: 0%;
                }
                .deploy-progress-details {
                    font-size: 13px;
                    color: #999;
                    margin-bottom: 15px;
                }
                .deploy-progress-log {
                    background: #0a0a0a;
                    border-radius: 4px;
                    padding: 10px;
                    max-height: 200px;
                    overflow-y: auto;
                    font-family: monospace;
                    font-size: 12px;
                }
                .deploy-progress-log-entry {
                    margin: 2px 0;
                    color: #999;
                }
                .deploy-progress-log-entry.error {
                    color: #f44336;
                }
                .deploy-progress-log-entry.success {
                    color: #4caf50;
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(this.container);

        // Close button
        this.container.querySelector('.deploy-progress-close').onclick = () => {
            this.minimize();
        };
    }

    _attachCallbacks() {
        this.monitor.onProgress = (event) => {
            this._handleProgress(event);
        };

        this.monitor.onError = (event) => {
            this._handleError(event);
        };

        this.monitor.onDone = (event) => {
            this._handleDone(event);
        };
    }

    _handleProgress(event) {
        const { type, stage, message, percent, speed, status, checkId } = event;

        if (type === 'check') {
            this._addCheck(checkId, message, status);
        } else if (type === 'progress') {
            this._updateProgress(stage, message, percent, speed);
        } else if (type === 'start') {
            this._addLog(message, 'info');
        }
    }

    _handleError(event) {
        const { message, suggestion } = event;
        this._addLog(`错误: ${message}`, 'error');
        if (suggestion) {
            this._addLog(`建议: ${suggestion}`, 'info');
        }
        this._setStage('部署失败', 'error');
    }

    _handleDone(event) {
        const { message, elapsed } = event;
        this._addLog(`${message} (耗时 ${elapsed}s)`, 'success');
        this._setStage('部署完成', 'success');
        this._updateProgress('done', '', 100);

        // Auto-close after 3 seconds
        setTimeout(() => this.close(), 3000);
    }

    _addCheck(checkId, message, status) {
        const checksContainer = this.container.querySelector('.deploy-progress-checks');
        const checkEl = document.createElement('div');
        checkEl.className = `deploy-progress-check ${status || ''}`;
        checkEl.textContent = message;
        checksContainer.appendChild(checkEl);
    }

    _updateProgress(stage, message, percent, speed) {
        this._setStage(message);

        if (percent !== undefined) {
            const bar = this.container.querySelector('.deploy-progress-bar-fill');
            bar.style.width = `${percent}%`;
        }

        const details = this.container.querySelector('.deploy-progress-details');
        const parts = [];
        if (percent !== undefined) {
            parts.push(`${percent.toFixed(1)}%`);
        }
        if (speed) {
            parts.push(speed);
        }
        details.textContent = parts.join(' · ');
    }

    _setStage(message, status) {
        const stageEl = this.container.querySelector('.deploy-progress-stage');
        stageEl.textContent = message;
        stageEl.style.color = status === 'error' ? '#f44336' :
                              status === 'success' ? '#4caf50' : '#e0e0e0';
    }

    _addLog(message, level = 'info') {
        const logContainer = this.container.querySelector('.deploy-progress-log');
        const entry = document.createElement('div');
        entry.className = `deploy-progress-log-entry ${level}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logContainer.appendChild(entry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    show() {
        this.container.style.display = 'flex';
        this.monitor.connect();
    }

    minimize() {
        this.container.style.display = 'none';
        // Keep monitor connected
    }

    close() {
        this.monitor.disconnect();
        if (this.container && this.container.parentNode) {
            this.container.parentNode.removeChild(this.container);
        }
    }
}

// ES6 module exports
export { DeployProgressMonitor, DeployProgressUI };
