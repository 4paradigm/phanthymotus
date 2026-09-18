/**
 * deploy-progress.js — deployment progress: one window, one code path
 *
 * `startDeploy()` is the only entry point the rest of the UI should use. It
 * owns the whole sequence — open the window, fire the POST, attach the socket
 * to the run the POST reports, drive the window to a terminal state — so the
 * top banner, the deploy panel and the solution loader can no longer drift into
 * showing different things for the same action. Before this, five call sites
 * each assembled their own combination of window / inline log / banner text,
 * and agent-core upgrades had a second, cut-down rendering of their own.
 *
 *   await startDeploy({ driverId, driverName, image });            // driver
 *   await startDeploy({ driverId, driverName, image, kind: 'core' });
 *
 * Event types on the wire:
 *   - start: Deployment started
 *   - check: Preflight check result (disk, network, registry)
 *   - progress: Pull/compose/start progress (has percent, speed fields)
 *   - error: Error occurred (has suggestion field)
 *   - done: Deployment complete
 */

class DeployProgressMonitor {
    /**
     * @param {string} driverId
     * @param {string} [runId] Run to follow, from the deploy POST response.
     *   The server replays that run from its start, so connecting after the
     *   deployment began — or after it already finished — still shows the whole
     *   sequence. Omit to follow whatever the driver is doing now.
     */
    constructor(driverId, runId = '') {
        this.driverId = driverId;
        this.runId = runId;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 3;
        this._closing = false;

        // Callbacks
        this.onProgress = null;
        this.onError = null;
        this.onDone = null;
        this.onConnected = null;
    }

    connect() {
        if (this._closing) return;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const token = localStorage.getItem('phanthy_access_token') || '';
        const run = this.runId ? `&run=${encodeURIComponent(this.runId)}` : '';
        const url = `${protocol}//${window.location.host}/api/ws/deploy/${this.driverId}?token=${token}${run}`;

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

            // Auto-reconnect on unexpected close. `_closing` keeps a close we
            // asked for from counting as unexpected — disconnect() used to
            // trigger the very reconnect loop it was meant to end, so every
            // finished deploy reopened its socket three more times.
            if (!this._closing && this.reconnectAttempts < this.maxReconnectAttempts) {
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
        this._closing = true;
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
        this.monitor = null;
        this.container = null;
        this.finished = false;
        this._lastLoggedPercent = -10; // Initialize to -10 so first log happens at 0%

        // Resolves when the window stops following this deployment — done,
        // error, or dismissed. Callers refresh what they show about the service
        // here: `startDeploy` returns as soon as the POST is accepted, which is
        // far too early to re-read a row (the old container is still running).
        this.settled = new Promise(resolve => { this._settle = resolve; });

        this._createUI();
    }

    /**
     * Follow a run over the deploy WebSocket. Call once the deploy POST has
     * returned its run id; the server replays that run from its first event, so
     * there is no race between opening the socket and starting the work.
     * @param {string} [runId]
     */
    attach(runId = '') {
        if (this.monitor) this.monitor.disconnect();
        this.monitor = new DeployProgressMonitor(this.driverId, runId);
        this._attachCallbacks();
        this.monitor.connect();
    }

    // ── Manual drive (no WebSocket) ──────────────────────────────────────
    // The same three events the monitor delivers, so a locally-observed step
    // (the core restart watcher below) renders through exactly the same code.

    pushProgress(message, percent, stage = 'update') {
        this._handleProgress({ type: 'progress', stage, message, percent });
    }

    pushLog(message, level = 'info') {
        this._addLog(message, level);
    }

    pushDone(event) {
        this._handleDone(event || {});
    }

    pushError(event) {
        this._handleError(typeof event === 'string' ? { message: event } : event);
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
        this.finished = true;
        this._settle({ ok: false, message });
        this._addLog(`错误: ${message}`, 'error');
        if (suggestion) {
            this._addLog(`建议: ${suggestion}`, 'info');
        }
        this._setStage('部署失败', 'error');
        if (this.monitor) this.monitor.disconnect();

        // Update minimized indicator
        const minimizedProgress = this.minimizedIndicator.querySelector('.deploy-progress-minimized-progress');
        if (minimizedProgress) {
            minimizedProgress.textContent = '部署失败';
            minimizedProgress.classList.add('error');
        }

        // A failed deploy stays on screen until it is dismissed. The failure
        // and its suggestion are the only place the operator can read what went
        // wrong; auto-closing this after 15s (and, worse, callers closing the
        // window themselves on a non-200) was how errors ended up invisible.
    }

    _handleDone(event) {
        const { message, elapsed } = event;
        this.finished = true;
        this._settle({ ok: true, message });
        // Only the deploy WebSocket carries `elapsed`; the core upgrade poll has
        // no equivalent. Unguarded this printed a literal "(耗时 undefineds)".
        const took = Number.isFinite(elapsed) ? ` (耗时 ${elapsed}s)` : '';
        this._addLog(`${message}${took}`, 'success');
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
        // Dismissing mid-deploy also settles: the caller's row is left showing
        // "部署中…" otherwise, and the deployment carries on server-side either
        // way (it is a background task, not tied to this socket).
        this._settle({ ok: this.finished, closed: true });
        if (this.monitor) this.monitor.disconnect();
        if (this.container && this.container.parentNode) {
            this.container.parentNode.removeChild(this.container);
        }
        if (this.minimizedIndicator && this.minimizedIndicator.parentNode) {
            this.minimizedIndicator.parentNode.removeChild(this.minimizedIndicator);
        }
    }
}

/**
 * The single way to start a deployment from the UI.
 *
 * Opens the progress window, fires the request, and attaches the window to the
 * run the server reports. Failures are shown in that window and nowhere else —
 * callers get a boolean and should not render the error a second time.
 *
 * @param {object}  opts
 * @param {string}  opts.driverId    Service id (also the progress channel).
 * @param {string}  opts.driverName  Title shown in the window.
 * @param {string}  [opts.image]     Target image; omit to use the manifest's.
 * @param {string}  [opts.kind]      'driver' (default) or 'core'.
 * @returns {Promise<{ok: boolean, ui: DeployProgressUI}>}
 */
async function startDeploy({ driverId, driverName, image = '', kind = 'driver' }) {
    const ui = new DeployProgressUI(driverId, driverName || driverId);
    ui.show();
    ui.pushProgress(kind === 'core' ? '正在启动升级…' : '正在提交部署…', 2);

    const url = kind === 'core'
        ? '/api/system/update'
        : `/api/drivers/${encodeURIComponent(driverId)}/deploy-v2`;
    const body = kind === 'core'
        ? { image, driver_id: driverId }
        : (image ? { image } : {});

    try {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const json = await res.json();
        if (json.code !== 200) {
            ui.pushError({ message: json.message || `请求失败（HTTP ${res.status}）` });
            return { ok: false, ui };
        }
        // Attach to the run this POST opened. Everything it has already emitted
        // is replayed, so a deploy that finished before the socket opened —
        // routine when the image is already local — still renders in full.
        ui.attach(json.data?.run_id || '');
        if (kind === 'core') _watchCoreRestart(driverId, image, ui);
        return { ok: true, ui };
    } catch (e) {
        ui.pushError({
            message: `网络错误: ${e.message}`,
            suggestion: '确认与设备的连接正常后重试。',
        });
        return { ok: false, ui };
    }
}

/**
 * Tail of an agent-core upgrade. The restart helper replaces this very process,
 * so the last thing the stream can report is "容器即将切换" — success is only
 * observable from outside, by reconnecting and finding the new tag. Reporting
 * it on reconnect alone would call a rolled-back upgrade a success.
 */
function _watchCoreRestart(driverId, image, ui) {
    const targetTag = (image || '').split(':').pop();
    let attempts = 0;
    let sawRestart = false;

    const timer = setInterval(async () => {
        attempts++;
        if (ui.finished) { clearInterval(timer); return; }

        try {
            const res = await fetch('/api/drivers');
            const json = await res.json();
            const entry = (json.data || []).find(d => d.id === driverId);
            const runningTag = (entry?.running_image || '').split(':').pop();

            if (sawRestart && runningTag && targetTag && runningTag === targetTag) {
                clearInterval(timer);
                ui.pushDone({ message: `升级完成：${runningTag}，页面即将刷新` });
                setTimeout(() => location.reload(), 2500);
            }
        } catch {
            // The API going away is the expected path to success here, not a
            // failure: agent-core restarts itself as the last act of the
            // upgrade. (The TLS cert changes with it, so the first requests
            // after it returns can fail too.)
            if (!sawRestart) {
                sawRestart = true;
                ui.pushProgress('服务重启中，等待重新连接…', 92, 'restart');
            }
        }

        if (attempts > 120) {   // 4 分钟
            clearInterval(timer);
            ui.pushError({
                message: '升级超时',
                suggestion: '容器可能仍在切换中，刷新页面查看当前版本。',
            });
        }
    }, 2000);
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
        if (!deployments.length) return;

        console.log('[DeployProgress] Found active deployments:', deployments);

        // Real service names, from the same list the deploy panel renders. The
        // hardcoded map this replaced knew three ids and showed every driver
        // its raw id.
        let names = {};
        try {
            const res = await fetch('/api/drivers', { headers: { 'Authorization': `Bearer ${token}` } });
            const json = await res.json();
            for (const d of (json.data || [])) names[d.id] = d.name || d.id;
        } catch { /* fall back to ids */ }

        for (const deployment of deployments) {
            const driverName = names[deployment.driver_id] || deployment.driver_id;

            console.log(`[DeployProgress] Restoring ${deployment.driver_id}...`);

            const progressWindow = new DeployProgressUI(deployment.driver_id, driverName);
            progressWindow.show();

            const elapsed = Math.floor(deployment.elapsed || 0);
            progressWindow._addLog(`重新连接到部署会话 (已运行 ${elapsed}s)`, 'info');
            progressWindow.attach(deployment.run_id || '');
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
export { DeployProgressMonitor, DeployProgressUI, startDeploy };
