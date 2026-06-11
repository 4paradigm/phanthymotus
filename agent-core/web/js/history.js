/**
 * history.js — 历史日志 Modal
 * 展示持久化的聊天会话列表 + 对话详情
 */

let _overlay, _list, _chat, _btnDeleteSelected, _selectedIds;
let _pollTimer = null;
let _activeSessionId = null;

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
  _btnDeleteSelected.addEventListener('click', deleteSelected);
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
  const html = turns.map(turn => turn.map(msg => _renderMessage(msg)).join('')).join('');
  _chat.innerHTML = `<div class="history-messages">${html}</div>`;
  _chat.scrollTop = _chat.scrollHeight;
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
