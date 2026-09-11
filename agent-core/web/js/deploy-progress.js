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
        const token = localStorage.getItem('phanthy_access_token') || '';
        const url = `${protocol}//${window.location.host}/api/ws/deploy/${this.driverId}?token=${token}`;

        console.log('[DeployProgress] Connecting to:', url);
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
            console.log(`[DeployProgress] Connected for driver ${this.driverId}`);
            this.reconnectAttempts = 0;
            if (this.onConnected) {
                this.onConnected();
            }
        };

        this.ws.onmessage = (event) => {
            console.log('[DeployProgress] Raw message:', event.data);
            try {
                const data = JSON.parse(event.data);
                this._handleMessage(data);
            } catch (e) {
                console.error('[DeployProgress] Failed to parse message:', e, event.data);
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
        console.log('[DeployProgress] Received message:', type, data);

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
        this._lastLoggedPercent = -10; // Initialize to -10 so first log happens at 0%

        this._createUI();
        this._attachCallbacks();
    }

    static _getMinimizedStack() {
        let stack = document.getElementById('deploy-progress-minimized-stack');
        if (!stack) {
            stack = document.createElement('div');
            stack.id = 'deploy-progress-minimized-stack';
            stack.className = 'deploy-progress-minimized-stack';
            document.body.appendChild(stack);
        }
        return stack;
    }

    _createUI() {
        // Create modal overlay
        this.container = document.createElement('div');
        this.container.className = 'deploy-progress-modal';
        this.container.innerHTML = `
            <div class="deploy-progress-content">
                <div class="deploy-progress-header">
                    <h3>部署进度：${this.driverName}</h3>
                    <div class="deploy-progress-header-actions">
                        <button class="deploy-progress-minimize" title="最小化">−</button>
                        <button class="deploy-progress-close" title="关闭">×</button>
                    </div>
                </div>
                <div class="deploy-progress-body">
                    <div class="deploy-progress-checks"></div>
                    <div class="deploy-progress-main">
                        <div class="deploy-progress-stage">正在连接...</div>
                        <div class="deploy-progress-bar">
                            <div class="deploy-progress-bar-fill"></div>
                        </div>
                        <div class="deploy-progress-details">等待部署开始</div>
                    </div>
                    <div class="deploy-progress-log"></div>
                </div>
            </div>
        `;

        // Create minimized indicator (hidden by default)
        this.minimizedIndicator = document.createElement('div');
        this.minimizedIndicator.className = 'deploy-progress-minimized hidden';
        this.minimizedIndicator.innerHTML = `
            <div class="deploy-progress-minimized-content">
                <div class="deploy-progress-minimized-title">
                    <span class="deploy-progress-minimized-spinner">⟳</span>
                    <span>部署中：${this.driverName}</span>
                </div>
                <div class="deploy-progress-minimized-progress"></div>
            </div>
        `;
        DeployProgressUI._getMinimizedStack().appendChild(this.minimizedIndicator);

        // Styles live in css/style.css (§ DEPLOY PROGRESS MODAL). They used to be
        // injected from here as a hardcoded dark palette, which is why this modal
        // rendered as a black terminal window inside a light parchment UI.

        document.body.appendChild(this.container);

        // Minimize button
        this.container.querySelector('.deploy-progress-minimize').onclick = () => {
            this.minimize();
        };

        // Close button
        this.container.querySelector('.deploy-progress-close').onclick = () => {
            this.close();
        };

        // Click minimized indicator to restore
        this.minimizedIndicator.onclick = () => {
            this.restore();
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
        const { type, stage, message, percent, speed, status, checkId, layer_id } = event;

        if (type === 'check') {
            this._addCheck(checkId, message, status);
            // Log checks (message already contains ✓/✗ from backend)
            this._addLog(message, status === 'fail' ? 'error' : 'success');
        } else if (type === 'layer') {
            // Log layer completion with progress if available
            const progressInfo = event.progress ? ` ${event.progress}` : '';
            this._addLog(`  ${layer_id}: ${status}${progressInfo}`, 'info');
        } else if (type === 'progress') {
            this._updateProgress(stage, message, percent, speed);
        } else if (type === 'start') {
            this._addLog(message, 'info');
        } else if (type === 'done') {
            this._addLog(message || '部署完成', 'success');
        }
    }

    _handleError(event) {
        const { message, suggestion } = event;
        this._addLog(`错误: ${message}`, 'error');
        if (suggestion) {
            this._addLog(`建议: ${suggestion}`, 'info');
        }
        this._setStage('部署失败', 'error');

        // Update minimized indicator
        const minimizedProgress = this.minimizedIndicator.querySelector('.deploy-progress-minimized-progress');
        if (minimizedProgress) {
            minimizedProgress.textContent = '部署失败';
            minimizedProgress.classList.add('error');
        }

        // Auto-close after 15 seconds
        setTimeout(() => this.close(), 15000);
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
        console.log('[DeployProgress] Updating progress:', { stage, message, percent, speed });
        // An empty message is not a stage — `_handleDone` calls this with '' just
        // to push the bar to 100%, and overwriting the stage with it wiped the
        // 部署完成 line the operator had just been shown.
        if (message) this._setStage(message);

        if (percent !== undefined) {
            const bar = this.container.querySelector('.deploy-progress-bar-fill');
            console.log('[DeployProgress] Setting bar width to:', percent + '%');
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

        // Add log entry for significant progress updates
        if (stage === 'pull' && percent !== undefined) {
            // Throttle pull progress logs (only log every 10%)
            const currentBucket = Math.floor(percent / 10);
            const lastBucket = Math.floor(this._lastLoggedPercent / 10);

            if (currentBucket > lastBucket) {
                const progressMsg = `${message} - ${percent.toFixed(1)}%` + (speed ? ` (${speed})` : '');
                this._addLog(progressMsg, 'info');
                this._lastLoggedPercent = percent;
            }
        } else if (stage !== 'pull' && message) {
            // Log all non-pull progress updates. Guarded on `message` for the
            // same reason as the stage above: the bar-to-100% call carries none,
            // and logging it left a bare timestamp as the last line of the log.
            this._addLog(message, 'info');
        }

        // Update minimized indicator
        const minimizedProgress = this.minimizedIndicator.querySelector('.deploy-progress-minimized-progress');
        if (minimizedProgress) {
            const progressText = percent !== undefined ? `${percent.toFixed(1)}%` : message || '进行中...';
            minimizedProgress.textContent = progressText;
        }
    }

    _setStage(message, status) {
        const stageEl = this.container.querySelector('.deploy-progress-stage');
        stageEl.textContent = message;
        // Class, not an inline colour: the theme owns green/red.
        stageEl.className = `deploy-progress-stage ${status || ''}`.trim();
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
        this.minimizedIndicator.classList.add('hidden');
        this.monitor.connect();
    }

    minimize() {
        this.container.style.display = 'none';
        this.minimizedIndicator.classList.remove('hidden');
        // Keep monitor connected
    }

    restore() {
        this.container.style.display = 'flex';
        this.minimizedIndicator.classList.add('hidden');
    }

    close() {
        this.monitor.disconnect();
        if (this.container && this.container.parentNode) {
            this.container.parentNode.removeChild(this.container);
        }
        if (this.minimizedIndicator && this.minimizedIndicator.parentNode) {
            this.minimizedIndicator.parentNode.removeChild(this.minimizedIndicator);
        }
    }
}

// Auto-restore active deployments on page load
async function restoreActiveDeployments() {
    try {
        const token = localStorage.getItem('phanthy_access_token') || '';
        const response = await fetch('/api/deploying', {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (!response.ok) {
            console.warn('[DeployProgress] Failed to fetch active deployments:', response.status);
            return;
        }

        const data = await response.json();
        const deployments = data.deployments || [];

        console.log('[DeployProgress] Found active deployments:', deployments);

        // Restore each active deployment
        for (const deployment of deployments) {
            const driverMap = {
                'perception': 'Perception Stack',
                'planning': 'Planning',
                'control': 'Control',
            };
            const driverName = driverMap[deployment.driver_id] || deployment.driver_id;

            console.log(`[DeployProgress] Restoring ${deployment.driver_id}...`);

            // Create progress window
            const progressWindow = new DeployProgressUI(deployment.driver_id, driverName);
            progressWindow.show();

            // Add a message indicating reconnection
            const elapsed = Math.floor(deployment.elapsed || 0);
            progressWindow._addLog(`重新连接到部署会话 (已运行 ${elapsed}s)`, 'info');
        }
    } catch (error) {
        console.error('[DeployProgress] Error restoring deployments:', error);
    }
}

// Run on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', restoreActiveDeployments);
} else {
    restoreActiveDeployments();
}

// ES6 module exports
export { DeployProgressMonitor, DeployProgressUI };
