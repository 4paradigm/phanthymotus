/**
 * history.js — 历史日志 Modal（会话记录 + 任务管理）
 */

let _overlay, _list, _chat, _btnDeleteSelected, _selectedIds;
let _pollTimer = null;
let _activeSessionId = null;
let _activeTab = 'sessions';

export function initHistory() {
  _overlay = document.getElementById('history-overlay');
  _list = document.getElementById('history-list');
  _chat = document.getElementById('history-chat');
  _btnDeleteSelected = document.getElementById('history-delete-selected');
  _selectedIds = new Set();

  document.getElementById('btn-history').addEventListener('click', showHistory);
  document.getElementById('history-close').addEventListener('click', hide);
  _overlay.addEventListener('click', e => { if (e.target === _overlay) hide(); });
  document.getElementById('history-clear-all').addEventListener('click', clearAll);
  document.getElementById('history-refresh').addEventListener('click', () => _refreshCurrentTab());
  _btnDeleteSelected.addEventListener('click', deleteSelected);

  // Tab switching
  _overlay.querySelectorAll('.history-tab').forEach(tab => {
    tab.addEventListener('click', () => _switchTab(tab.dataset.tab));
  });
}

export async function showHistory() {
  _overlay.classList.remove('hidden');
  _selectedIds.clear();
  _updateDeleteBtn();
  await _loadSessions();
  // 打开时每 5 秒自动刷新 session 列表
  _startPoll();
}

function hide() {
  _overlay.classList.add('hidden');
  _stopPoll();
}

function _startPoll() {
  _stopPoll();
  _pollTimer = setInterval(() => _loadSessions(), 5000);
}

function _stopPoll() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

async function _loadSessions() {
  try {
    const res = await fetch('/api/history/sessions');
    const data = await res.json();
    _renderList(data.sessions);
  } catch (e) {
    _list.innerHTML = '<div class="history-empty">加载失败</div>';
  }
}

function _renderList(sessions) {
  if (!sessions.length) {
    _list.innerHTML = '<div class="history-empty">暂无对话记录</div>';
    return;
  }
  _list.innerHTML = sessions.map(s => `
    <div class="history-session-item${s.id === _activeSessionId ? ' active' : ''}" data-id="${s.id}">
      <label class="history-session-check">
        <input type="checkbox" class="history-cb" data-id="${s.id}"${_selectedIds.has(s.id) ? ' checked' : ''}>
      </label>
      <div class="history-session-info">
        <div class="history-session-summary">${_escape(s.summary || '(无标题)')}</div>
        <div class="history-session-meta">
          <span>${_formatTime(s.started_at)}</span>
          <span>${s.turn_count} 轮</span>
        </div>
      </div>
    </div>
  `).join('');

  // Click to view
  _list.querySelectorAll('.history-session-info').forEach(el => {
    el.addEventListener('click', () => {
      const item = el.closest('.history-session-item');
      _list.querySelectorAll('.history-session-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      _loadSession(item.dataset.id);
    });
  });

  // Checkbox selection
  _list.querySelectorAll('.history-cb').forEach(cb => {
    cb.addEventListener('change', () => {
      if (cb.checked) _selectedIds.add(cb.dataset.id);
      else _selectedIds.delete(cb.dataset.id);
      _updateDeleteBtn();
    });
  });
}

function _updateDeleteBtn() {
  _btnDeleteSelected.disabled = _selectedIds.size === 0;
  _btnDeleteSelected.textContent = _selectedIds.size ? `删除选中 (${_selectedIds.size})` : '删除选中';
}

async function deleteSelected() {
  if (!_selectedIds.size) return;
  if (!confirm(`确认删除 ${_selectedIds.size} 条记录？`)) return;
  await fetch('/api/history/sessions/batch-delete', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ids: [..._selectedIds] }),
  });
  _selectedIds.clear();
  _updateDeleteBtn();
  _chat.innerHTML = '<div class="history-placeholder">选择一个会话查看对话记录</div>';
  await _loadSessions();
}

async function clearAll() {
  if (_activeTab === 'tasks') {
    if (!confirm('确认清除所有活跃任务？')) return;
    await fetch('/api/tasks', { method: 'DELETE' });
    _loadTasks();
    return;
  }
  if (!confirm('确认清空全部历史记录？此操作不可恢复。')) return;
  await fetch('/api/history/sessions', { method: 'DELETE' });
  _selectedIds.clear();
  _updateDeleteBtn();
  _chat.innerHTML = '<div class="history-placeholder">选择一个会话查看对话记录</div>';
  await _loadSessions();
}

async function _loadSession(sessionId) {
  _activeSessionId = sessionId;
  _chat.innerHTML = '<div class="history-placeholder">加载中…</div>';
  try {
    const res = await fetch(`/api/history/sessions/${sessionId}`);
    const data = await res.json();
    _renderChat(data.messages);
  } catch (e) {
    _chat.innerHTML = '<div class="history-placeholder">加载失败</div>';
  }
}

function _renderChat(turns) {
  if (!turns.length) {
    _chat.innerHTML = '<div class="history-placeholder">此会话无消息</div>';
    return;
  }
  const summaryHtml = _renderUsageSummary(turns);
  const html = turns.map(turn => {
    const msgs = turn.map(msg => _renderMessage(msg)).join('');
    const usage = _extractTurnUsage(turn);
    const usageHtml = usage
      ? `<div class="history-turn-divider">
           <span class="history-usage">输入 ${_fmtTokens(usage.prompt_tokens)} · 输出 ${_fmtTokens(usage.completion_tokens)} · 缓存 ${_fmtTokens(usage.cached_tokens)}</span>
         </div>`
      : '<div class="history-turn-divider"></div>';
    return msgs + usageHtml;
  }).join('');
  _chat.innerHTML = `<div class="history-messages">${summaryHtml}${html}</div>`;
  _chat.scrollTop = _chat.scrollHeight;
}

function _extractTurnUsage(turn) {
  for (const msg of turn) {
    if (msg._usage) return msg._usage;
  }
  return null;
}

function _renderUsageSummary(turns) {
  let totalPrompt = 0, totalCompletion = 0, totalCached = 0;
  for (const turn of turns) {
    const u = _extractTurnUsage(turn);
    if (u) {
      totalPrompt += u.prompt_tokens || 0;
      totalCompletion += u.completion_tokens || 0;
      totalCached += u.cached_tokens || 0;
    }
  }
  if (!totalPrompt && !totalCompletion) return '';
  return `<div class="history-usage-summary">
    会话用量: 输入 ${_fmtTokens(totalPrompt)} · 输出 ${_fmtTokens(totalCompletion)} · 缓存 ${_fmtTokens(totalCached)} tokens
  </div>`;
}

function _fmtTokens(n) {
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(1) + 'G';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 10_000) return (n / 1_000).toFixed(1) + 'K';
  if (n >= 1_000) return (n / 1_000).toFixed(2) + 'K';
  return String(n);
}

function _renderMessage(msg) {
  if (msg.role === 'user') {
    return `<div class="history-msg history-msg-user">${_renderContent(msg.content)}</div>`;
  }
  if (msg.role === 'assistant') {
    let html = '';
    // Text content
    const text = _extractText(msg.content);
    if (text) {
      html += `<div class="history-msg history-msg-assistant">${_escape(text)}</div>`;
    }
    // Tool calls
    if (msg.tool_calls && msg.tool_calls.length) {
      html += msg.tool_calls.map(tc => _renderToolCall(tc)).join('');
    }
    return html;
  }
  if (msg.role === 'tool') {
    return _renderToolResult(msg);
  }
  return '';
}

function _renderToolCall(tc) {
  const name = tc.function?.name || 'unknown';
  let args = tc.function?.arguments || '';
  try { args = JSON.stringify(JSON.parse(args), null, 2); } catch {}
  return `
    <details class="history-tool-card history-tool-call">
      <summary><span class="history-tool-icon">⚡</span> ${_escape(name)}</summary>
      <pre class="history-tool-body">${_escape(args)}</pre>
    </details>
  `;
}

function _renderToolResult(msg) {
  const content = msg.content || '';
  // Try to find the tool name from tool_call_id context (not available here, use generic label)
  let display = content;
  try {
    const parsed = JSON.parse(content);
    display = JSON.stringify(parsed, null, 2);
  } catch {}
  return `
    <details class="history-tool-card history-tool-result">
      <summary><span class="history-tool-icon">📋</span> 执行结果</summary>
      <pre class="history-tool-body">${_escape(display)}</pre>
    </details>
  `;
}

function _renderContent(content) {
  if (typeof content === 'string') return _escape(content);
  if (Array.isArray(content)) {
    return content.map(part => {
      if (part.type === 'text') return _escape(part.text || '');
      if (part.type === 'image_url') return '<span class="history-img-tag">[图片]</span>';
      return '';
    }).join('');
  }
  return '';
}

function _extractText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.filter(p => p.type === 'text').map(p => p.text).join('');
  }
  return '';
}

function _escape(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function _formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}


// ── Tab Switching ────────────────────────────────────────────────────────────

function _switchTab(tab) {
  _activeTab = tab;
  _overlay.querySelectorAll('.history-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tab);
  });
  const sessionsPane = _overlay.querySelector('.history-sessions-pane');
  const tasksPane = document.getElementById('history-tasks');
  const deleteBtn = _btnDeleteSelected;
  const clearBtn = document.getElementById('history-clear-all');

  if (tab === 'sessions') {
    sessionsPane.classList.remove('hidden');
    tasksPane.classList.add('hidden');
    deleteBtn.classList.remove('hidden');
    clearBtn.textContent = '清空全部';
  } else {
    sessionsPane.classList.add('hidden');
    tasksPane.classList.remove('hidden');
    deleteBtn.classList.add('hidden');
    clearBtn.textContent = '清除所有任务';
    _loadTasks();
  }
}

function _refreshCurrentTab() {
  if (_activeTab === 'sessions') _loadSessions();
  else _loadTasks();
}


// ── Tasks Tab ────────────────────────────────────────────────────────────────

async function _loadTasks() {
  const el = document.getElementById('history-tasks');
  el.innerHTML = '<div class="history-empty">加载中…</div>';
  try {
    const res = await fetch('/api/tasks');
    const data = await res.json();
    _renderTasks(el, data.tasks || []);
  } catch {
    el.innerHTML = '<div class="history-empty">加载失败</div>';
  }
}

function _renderTasks(el, tasks) {
  if (!tasks.length) {
    el.innerHTML = '<div class="history-empty">当前没有活跃任务</div>';
    return;
  }

  el.innerHTML = tasks.map(t => {
    const elapsed = _elapsedStr(t.created_at);
    return `
      <div class="task-card" data-id="${t.id}">
        <div class="task-header">
          <span class="task-goal">${_escape(t.goal)}</span>
          <span class="task-badge ${t.status}">${t.status}</span>
        </div>
        <div class="task-meta">
          ${t.progress ? `<div>进度: ${_escape(t.progress)}</div>` : ''}
          ${t.check_cron ? `<div>定时: ${_escape(t.check_cron)}</div>` : ''}
          <div>${elapsed} · ID: ${t.id}</div>
        </div>
        <div class="task-actions">
          <button class="task-edit-btn" data-id="${t.id}">编辑</button>
          <button class="task-done-btn" data-id="${t.id}">标记完成</button>
          <button class="task-delete-btn danger" data-id="${t.id}">删除</button>
        </div>
      </div>`;
  }).join('');

  // Bind actions
  el.querySelectorAll('.task-done-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      await fetch(`/api/tasks/${btn.dataset.id}/done`, { method: 'POST' });
      _loadTasks();
    });
  });
  el.querySelectorAll('.task-delete-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      if (!confirm('确认删除此任务？')) return;
      await fetch(`/api/tasks/${btn.dataset.id}`, { method: 'DELETE' });
      _loadTasks();
    });
  });
  el.querySelectorAll('.task-edit-btn').forEach(btn => {
    btn.addEventListener('click', () => _showTaskEdit(btn.dataset.id, tasks));
  });
}

function _showTaskEdit(taskId, tasks) {
  const task = tasks.find(t => t.id === taskId);
  if (!task) return;
  const card = document.querySelector(`.task-card[data-id="${taskId}"]`);
  if (!card) return;

  // Replace card content with edit form
  card.innerHTML = `
    <div class="task-edit-form">
      <label>目标</label>
      <input type="text" class="task-input" id="edit-goal-${taskId}" value="${_escape(task.goal)}">
      <label>进度</label>
      <textarea class="task-input" id="edit-progress-${taskId}" rows="2">${_escape(task.progress || '')}</textarea>
      <label>定时 Cron</label>
      <input type="text" class="task-input" id="edit-cron-${taskId}" value="${_escape(task.check_cron || '')}" placeholder="如 */5 * * * *">
      <div class="task-actions" style="margin-top:10px">
        <button class="task-save-btn">保存</button>
        <button class="task-cancel-btn">取消</button>
      </div>
    </div>`;

  card.querySelector('.task-save-btn').addEventListener('click', async () => {
    const body = {
      goal: document.getElementById(`edit-goal-${taskId}`).value,
      progress: document.getElementById(`edit-progress-${taskId}`).value,
      check_cron: document.getElementById(`edit-cron-${taskId}`).value,
    };
    await fetch(`/api/tasks/${taskId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    _loadTasks();
  });
  card.querySelector('.task-cancel-btn').addEventListener('click', () => _loadTasks());
}

function _elapsedStr(createdAt) {
  const elapsed = (Date.now() / 1000) - createdAt;
  if (elapsed < 60) return `${Math.floor(elapsed)}s`;
  if (elapsed < 3600) return `${Math.floor(elapsed / 60)}min`;
  return `${(elapsed / 3600).toFixed(1)}h`;
}
