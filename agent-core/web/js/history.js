/**
 * history.js — 历史日志 Modal（会话记录 + 任务管理 + 运行时定时）
 */

let _overlay, _list, _chat, _btnDeleteSelected, _selectedIds, _btnBack, _detailTitle;
let _pollTimer = null;
let _activeSessionId = null;
let _activeTab = 'sessions';
let _scheduleSettings = {};
let _scheduleLoadId = 0;
// 上一次渲染的内容指纹。轮询每 3 秒拉一次，内容没变就别重画 —— 重画会清掉
// 滚动位置和展开的工具卡片。
let _listSig = '';
let _chatSig = '';

const POLL_MS = 3000;

const KIND_GROUPS = [
  { kind: 'main',        label: '主代理' },
  { kind: 'subagent',    label: '子代理' },
  { kind: 'bg_subagent', label: '后台子代理' },
];

export function initHistory() {
  _overlay = document.getElementById('history-overlay');
  _list = document.getElementById('history-list');
  _chat = document.getElementById('history-chat');
  _btnDeleteSelected = document.getElementById('history-delete-selected');
  _btnBack = document.getElementById('history-back');
  _detailTitle = document.getElementById('history-detail-title');
  _selectedIds = new Set();

  document.getElementById('btn-history').addEventListener('click', showHistory);
  document.getElementById('history-close').addEventListener('click', hide);
  _overlay.addEventListener('click', e => { if (e.target === _overlay) hide(); });
  document.getElementById('history-clear-all').addEventListener('click', clearAll);
  document.getElementById('history-refresh').addEventListener('click', () => _refreshCurrentTab());
  _btnDeleteSelected.addEventListener('click', deleteSelected);
  _btnBack.addEventListener('click', _closeDetail);

  // Tab switching
  _overlay.querySelectorAll('.history-tab').forEach(tab => {
    tab.addEventListener('click', () => _switchTab(tab.dataset.tab));
  });
}

export async function showHistory() {
  _overlay.classList.remove('hidden');
  _selectedIds.clear();
  _updateDeleteBtn();
  await _refreshCurrentTab();
  _startPoll();
}

function hide() {
  _overlay.classList.add('hidden');
  _closeDetail();
  _stopPoll();
}

function _startPoll() {
  _stopPoll();
  // 列表和当前打开的会话一起刷 —— 以前只刷列表，右侧停在打开那一刻的快照，
  // 一个还在跑的 turn 要等下次手动点开才看得到。
  _pollTimer = setInterval(() => {
    if (_activeTab === 'sessions') { _loadSessions(); _refreshOpenSession(); }
    else if (_activeTab === 'tasks') _loadTasks();
    // 定时页只在切换、刷新或操作后加载，避免轮询清掉正在填写的表单。
  }, POLL_MS);
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
    _list.innerHTML = '<div class="history-empty">读不到会话列表。确认机器在线后点「刷新」重试。</div>';
    _listSig = '';
  }
}

/** 判定一条记录的来源。
 *
 * summary 的前缀优先于 kind 列：kind 是后加的列，迁移时给所有老记录填了默认值
 * 'main'，所以库里已有的子代理记录都自称主代理。`[subagent:` 这个前缀只有子代理
 * 的落盘路径会写，用它兜底比相信老行的 kind 准。
 */
function _sessionKind(s) {
  const sum = s.summary || '';
  if (sum.startsWith('[subagent:')) {
    return sum.includes('[bg]') ? 'bg_subagent' : 'subagent';
  }
  return s.kind || 'main';
}

/** 列表标题：摘掉分区和 id 标签已经表达过的前缀，以及 `<event …>` 信封。
 *
 * 信封现在由后端 `summary_text` 在存之前就剥掉了，这里处理的是那之前存下的老记录。
 * 它们存的是截断在 100 字的信封本身，而信封的开标签本身就可能超过 100 字 ——
 * 天轶上那条就是：`<event source="dds:/nvidia_desktop/ext_mic/card_…/audio/asr"
 * channel="local_mic" ts="2026…`，连 `>` 都没截到。所以不能指望能匹配到完整标签，
 * 按「有没有 `>`」分情况，正文取不到就退回显示 source —— 从哪个渠道进来的，是这行
 * 里仅存的有用信息。
 */
function _sessionTitle(s) {
  const raw = (s.summary || '')
    .replace(/^\[subagent:[^\]]+\]\s*/, '')
    .replace(/^\[bg\]\s*/, '')
    .trim();
  if (!raw.startsWith('<event')) return raw || '(无标题)';

  const close = raw.indexOf('>');
  const body = close >= 0
    ? raw.slice(close + 1).replace(/<\/?event\b[^>]*>/g, '').trim()
    : '';
  if (body) {
    // 正文常常是 `{"text": "早上好。", "audio_duration_ms": 1676}`，可能也被截断了。
    const text = body.match(/"text"\s*:\s*"((?:[^"\\]|\\.)*)"/);
    if (text) return text[1];
    if (!body.startsWith('{')) return body;
  }
  const source = raw.match(/source="([^"]+)"/);
  return source ? `来自 ${source[1]}` : '(无标题)';
}

function _sessionAgentId(s) {
  const m = (s.summary || '').match(/^\[subagent:([^\]]+)\]/);
  return m ? m[1] : '';
}

function _renderList(sessions) {
  if (!sessions.length) {
    _list.innerHTML = '<div class="history-empty">还没有对话。代理收到消息或事件后，会话会出现在这里。</div>';
    _listSig = '';
    return;
  }

  // 后端已按最后活动时间倒序返回，分组时保持该顺序即可。
  const byKind = { main: [], subagent: [], bg_subagent: [] };
  for (const s of sessions) (byKind[_sessionKind(s)] || byKind.main).push(s);

  const sig = JSON.stringify([
    _activeSessionId, [..._selectedIds].sort(),
    sessions.map(s => [s.id, s.turn_count, s.last_at, s.summary]),
  ]);
  if (sig === _listSig) return;
  _listSig = sig;

  const scrollTop = _list.scrollTop;
  _list.innerHTML = KIND_GROUPS.map(({ kind, label }) => {
    const group = byKind[kind];
    if (!group.length) return '';
    const items = group.map(s => {
      const agentId = _sessionAgentId(s);
      return `
      <div class="history-session-item kind-${kind}${s.id === _activeSessionId ? ' active' : ''}" data-id="${s.id}">
        <label class="history-session-check">
          <input type="checkbox" class="history-cb" data-id="${s.id}"${_selectedIds.has(s.id) ? ' checked' : ''}>
        </label>
        <div class="history-session-info">
          <div class="history-session-summary">${_escape(_sessionTitle(s))}</div>
          <div class="history-session-meta">
            <span>${_formatTime(s.last_at || s.started_at)}</span>
            <span>${s.turn_count} 轮</span>
            ${agentId ? `<span class="history-session-agent">${_escape(agentId)}</span>` : ''}
          </div>
        </div>
      </div>`;
    }).join('');
    return `<div class="history-group">
      <div class="history-group-head">${label}<span class="history-group-count">${group.length}</span></div>
      ${items}
    </div>`;
  }).join('');
  _list.scrollTop = scrollTop;

  // Click to view
  _list.querySelectorAll('.history-session-info').forEach(el => {
    el.addEventListener('click', () => {
      const item = el.closest('.history-session-item');
      _list.querySelectorAll('.history-session-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      _loadSession(item.dataset.id, el.querySelector('.history-session-summary').textContent);
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
  _clearChatPane();
  await _loadSessions();
}

function _clearChatPane() {
  _closeDetail();
  _activeSessionId = null;
  _chatSig = '';
  _chat.innerHTML = '<div class="history-placeholder">选择一个会话查看对话记录</div>';
}

async function clearAll() {
  if (_activeTab === 'schedules') return;
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
  _clearChatPane();
  await _loadSessions();
}

async function _loadSession(sessionId, title = '') {
  _activeSessionId = sessionId;
  _chatSig = '';
  _chat.innerHTML = '<div class="history-placeholder">加载中…</div>';
  _openDetail(title);
  await _fetchSession(sessionId);
}

/* 手机上列表和对话各占满一屏，点开一条会话是「推进」到详情，而不是把两个都压扁。
   两个 pane 在桌面端并排不变 —— 这几个类只在 768px 以下有布局效果。 */
function _openDetail(title) {
  _overlay.classList.add('showing-detail');
  _detailTitle.textContent = title;
}

function _closeDetail() {
  _overlay.classList.remove('showing-detail');
  _detailTitle.textContent = '';
}

/** 轮询时刷新右侧，但保留滚动位置和展开的卡片。 */
async function _refreshOpenSession() {
  if (_activeSessionId) await _fetchSession(_activeSessionId);
}

async function _fetchSession(sessionId) {
  try {
    const res = await fetch(`/api/history/sessions/${sessionId}`);
    const data = await res.json();
    if (sessionId !== _activeSessionId) return;  // 期间切走了
    _renderChat(data.messages, data.turn_times || []);
  } catch (e) {
    if (!_chatSig) _chat.innerHTML = '<div class="history-placeholder">读不到这段对话。确认机器在线后点「刷新」重试。</div>';
  }
}

function _renderChat(turns, turnTimes) {
  if (!turns.length) {
    _chat.innerHTML = '<div class="history-placeholder">此会话无消息</div>';
    _chatSig = '';
    return;
  }
  const sig = JSON.stringify([_activeSessionId, turns, turnTimes]);
  if (sig === _chatSig) return;
  const firstRender = !_chatSig;
  _chatSig = sig;

  // 重画前记住位置：贴着底的会话继续贴底（新轮次自动进入视野），
  // 往回翻过的保持原处，否则每 3 秒把人弹回底部。
  const atBottom = firstRender ||
    (_chat.scrollHeight - _chat.scrollTop - _chat.clientHeight) < 40;
  const prevTop = _chat.scrollTop;
  const openCards = new Set();
  _chat.querySelectorAll('details[open]').forEach(d => openCards.add(d.dataset.key));

  const summaryHtml = _renderUsageSummary(turns);
  const html = turns.map((turn, i) => {
    const msgs = turn.map((msg, j) => _renderMessage(msg, `${i}-${j}`, openCards)).join('');
    const usage = _extractTurnUsage(turn);
    const t = turnTimes[i];
    const timeHtml = t
      ? `<span class="history-turn-time">${_formatDateTime(t.updated_at || t.started_at)}</span>`
      : '<span></span>';
    const usageHtml = usage
      ? `<span class="history-usage">输入 ${_fmtTokens(usage.prompt_tokens)} · 输出 ${_fmtTokens(usage.completion_tokens)} · 缓存 ${_fmtTokens(usage.cached_tokens)}</span>`
      : '';
    return msgs + `<div class="history-turn-divider">${timeHtml}${usageHtml}</div>`;
  }).join('');
  _chat.innerHTML = `<div class="history-messages">${summaryHtml}${html}</div>`;
  _chat.scrollTop = atBottom ? _chat.scrollHeight : prevTop;
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

function _renderMessage(msg, key = '', openCards = new Set()) {
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
      html += msg.tool_calls.map((tc, k) => _renderToolCall(tc, `${key}-c${k}`, openCards)).join('');
    }
    return html;
  }
  if (msg.role === 'tool') {
    return _renderToolResult(msg, `${key}-r`, openCards);
  }
  return '';
}

function _renderToolCall(tc, key = '', openCards = new Set()) {
  const name = tc.function?.name || 'unknown';
  let args = tc.function?.arguments || '';
  try { args = JSON.stringify(JSON.parse(args), null, 2); } catch {}
  return `
    <details class="history-tool-card history-tool-call" data-key="${key}"${openCards.has(key) ? ' open' : ''}>
      <summary><span class="history-tool-icon">⚡</span> ${_escape(name)}</summary>
      <pre class="history-tool-body">${_escape(args)}</pre>
    </details>
  `;
}

function _renderToolResult(msg, key = '', openCards = new Set()) {
  const content = msg.content || '';
  // Try to find the tool name from tool_call_id context (not available here, use generic label)
  let display = content;
  try {
    const parsed = JSON.parse(content);
    display = JSON.stringify(parsed, null, 2);
  } catch {}
  return `
    <details class="history-tool-card history-tool-result" data-key="${key}"${openCards.has(key) ? ' open' : ''}>
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
  return d.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// 一律按北京时间显示：机器人分布在不同机器上，看日志的人和机器人的时区不一定一致，
// 用浏览器本地时区读出来的时间没法和机器上的日志对齐。
const _BJ_PARTS = new Intl.DateTimeFormat('en-GB', {
  timeZone: 'Asia/Shanghai',
  year: 'numeric', month: '2-digit', day: '2-digit',
  hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
});

function _bjParts(ts) {
  const p = {};
  for (const { type, value } of _BJ_PARTS.formatToParts(new Date(ts * 1000))) p[type] = value;
  return p;
}

/** 完整日期时间，用于每一轮的时间戳。 */
function _formatDateTime(ts) {
  if (!ts) return '';
  const p = _bjParts(ts);
  return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second}`;
}

/** 列表里的紧凑形式：今年的记录省掉年份。 */
function _formatTime(ts) {
  if (!ts) return '';
  const p = _bjParts(ts);
  const thisYear = _bjParts(Date.now() / 1000).year;
  const date = p.year === thisYear ? `${p.month}-${p.day}` : `${p.year}-${p.month}-${p.day}`;
  return `${date} ${p.hour}:${p.minute}`;
}


// ── Tab Switching ────────────────────────────────────────────────────────────

function _switchTab(tab) {
  _activeTab = tab;
  _closeDetail();
  _overlay.querySelectorAll('.history-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tab);
  });
  _overlay.querySelector('.history-sessions-pane').classList.toggle('hidden', tab !== 'sessions');
  document.getElementById('history-tasks').classList.toggle('hidden', tab !== 'tasks');
  document.getElementById('history-schedules').classList.toggle('hidden', tab !== 'schedules');
  _btnDeleteSelected.classList.toggle('hidden', tab !== 'sessions');
  const clearBtn = document.getElementById('history-clear-all');
  clearBtn.classList.toggle('hidden', tab === 'schedules');

  if (tab === 'sessions') {
    clearBtn.textContent = '清空全部';
  } else if (tab === 'tasks') {
    clearBtn.textContent = '清除所有任务';
    _loadTasks();
  } else if (tab === 'schedules') {
    _loadSchedules();
  }
}

function _refreshCurrentTab() {
  if (_activeTab === 'sessions') return _loadSessions();
  if (_activeTab === 'tasks') return _loadTasks();
  if (_activeTab === 'schedules') return _loadSchedules();
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
    el.innerHTML = '<div class="history-empty">读不到任务列表。确认机器在线后点「刷新」重试。</div>';
  }
}

function _renderTasks(el, tasks) {
  if (!tasks.length) {
    el.innerHTML = '<div class="history-empty">没有进行中的任务。代理接到长期目标时会在这里建任务并跟踪进度。</div>';
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
          <div class="task-meta-foot">已运行 ${elapsed}<span class="task-id">${t.id}</span></div>
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

// ── Schedules Tab ────────────────────────────────────────────────────────────
// 运行时定时任务（/api/schedules）。到点把文字推给机器人、叫醒它跑一轮，所以页面上
// 一直提醒别设得太频繁；后端的最短间隔 / 每日上限会拒绝过密的设置，错误原样显示。

const _CREATOR_TEXT = { web: '网页建', api: '接口建', model: '模型建' };

async function _scheduleFetch(url, method = 'GET', body = null) {
  const opts = { method };
  if (body !== null) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  return res.json();
}

async function _loadSchedules() {
  const el = document.getElementById('history-schedules');
  const loadId = ++_scheduleLoadId;
  if (!el.innerHTML) el.innerHTML = '<div class="history-empty">加载中…</div>';
  try {
    const data = await _scheduleFetch('/api/schedules');
    if (loadId !== _scheduleLoadId) return;
    _scheduleSettings = data.settings || {};
    _renderSchedules(el, data);
  } catch {
    if (loadId !== _scheduleLoadId) return;
    el.innerHTML = '<div class="history-empty">读不到定时任务。确认机器在线后点「刷新」重试。</div>';
  }
}

function _scheduleWhen(s) {
  const tz = _escape(s.tz || '容器本地时区');
  return s.kind === 'once'
    ? `一次：${_escape(s.at)}（${tz}）`
    : `每次：${_escape(s.cron)}（${tz}）`;
}

function _renderSchedules(el, data) {
  const list = data.schedules || [];
  const jobs = data.config_jobs || [];
  const st = _scheduleSettings;

  const cards = list.map(s => `
      <div class="task-card" data-id="${_escape(s.id)}">
        <div class="task-header">
          <span class="task-goal">${_escape(s.name)}</span>
          <span class="task-badge ${s.enabled ? 'running' : 'paused'}">${s.enabled ? '启用' : '停用'}</span>
        </div>
        <div class="task-meta">
          <div>${_scheduleWhen(s)}</div>
          <div>下次：${_escape(s.next_run_text || '无')}　上次：${_escape(s.last_run_text || '从未')}　已触发 ${s.fire_count || 0} 次</div>
          ${s.skill ? `<div>技能：${_escape(s.skill)}</div>` : ''}
          <div>文字：${_escape(s.text)}</div>
          <div class="task-meta-foot">${_CREATOR_TEXT[s.created_by] || _escape(s.created_by)}<span class="task-id">${_escape(s.id)}</span></div>
        </div>
        <div class="task-actions">
          <button class="sched-edit-btn" data-id="${_escape(s.id)}">编辑</button>
          <button class="sched-toggle-btn" data-id="${_escape(s.id)}">${s.enabled ? '停用' : '启用'}</button>
          <button class="sched-run-btn" data-id="${_escape(s.id)}" ${s.enabled ? '' : 'disabled title="先启用再触发"'}>立即触发</button>
          <button class="sched-delete-btn danger" data-id="${_escape(s.id)}">删除</button>
        </div>
      </div>`).join('');

  const jobCards = jobs.map(j => `
      <div class="task-card">
        <div class="task-header"><span class="task-goal">${_escape(j.name)}</span></div>
        <div class="task-meta">
          <div>每次：${_escape(j.cron)}（${_escape(j.tz || '容器本地时区')}）</div>
          <div>下次：${_escape(j.next_run_text || '无')}</div>
          <div>文字：${_escape(j.text)}</div>
          ${j.error ? `<div class="schedule-error">${_escape(j.error)}</div>` : ''}
        </div>
      </div>`).join('');

  el.innerHTML = `
    <div class="schedule-tip">到点会把文字推给机器人，叫醒它做事。每次触发都要跑一轮模型，请不要设得太频繁。</div>
    <details class="schedule-settings">
      <summary>设置</summary>
      <label class="schedule-check"><input type="checkbox" id="sched-set-model" ${st.model_tools_enabled ? 'checked' : ''}> 允许模型自己建定时（默认关）</label>
      <label>最短间隔（分钟）<input type="number" min="1" class="task-input" id="sched-set-interval" value="${_escape(String(st.min_interval_minutes ?? ''))}"></label>
      <label>每天最多触发次数<input type="number" min="1" class="task-input" id="sched-set-fires" value="${_escape(String(st.max_fires_per_day ?? ''))}"></label>
      <label>最多条数<input type="number" min="1" class="task-input" id="sched-set-max" value="${_escape(String(st.max_schedules ?? ''))}"></label>
      <label>每个技能最多条数<input type="number" min="0" class="task-input" id="sched-set-skill" value="${_escape(String(st.max_per_skill ?? ''))}"></label>
      <label>默认时区<input type="text" class="task-input" id="sched-set-tz" value="${_escape(st.default_tz || '')}"></label>
      <div class="task-actions"><button id="sched-set-save">保存设置</button><span class="schedule-msg" id="sched-set-msg"></span></div>
    </details>
    <div class="task-actions"><button class="task-edit-btn" id="schedule-new-btn">新建定时任务</button></div>
    <div id="schedule-form-slot"></div>
    ${cards || '<div class="history-empty">还没有定时任务。</div>'}
    ${jobs.length ? `<div class="schedule-section-title">配置文件里的定时（只读，改要改配置并重启 agent-core）</div>${jobCards}` : ''}`;

  el.querySelector('#schedule-new-btn').addEventListener('click', () => _openScheduleForm(null));
  el.querySelector('#sched-set-save').addEventListener('click', _saveScheduleSettings);

  const byId = id => list.find(s => s.id === id);
  el.querySelectorAll('.sched-edit-btn').forEach(btn => {
    btn.addEventListener('click', () => _openScheduleForm(byId(btn.dataset.id)));
  });
  el.querySelectorAll('.sched-toggle-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const s = byId(btn.dataset.id);
      const r = await _scheduleFetch(`/api/schedules/${encodeURIComponent(s.id)}`, 'PATCH', { enabled: !s.enabled });
      if (!r.ok) alert(r.error || '操作失败');
      _loadSchedules();
    });
  });
  el.querySelectorAll('.sched-run-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      if (!confirm('现在触发一次？机器人会被叫醒跑一轮，算进今天的触发次数。')) return;
      const r = await _scheduleFetch(`/api/schedules/${encodeURIComponent(btn.dataset.id)}/run`, 'POST');
      if (!r.ok) alert(r.error || '触发失败');
      _loadSchedules();
    });
  });
  el.querySelectorAll('.sched-delete-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      if (!confirm('确认删除这个定时任务？')) return;
      const r = await _scheduleFetch(`/api/schedules/${encodeURIComponent(btn.dataset.id)}`, 'DELETE');
      if (!r.ok) alert(r.error || '删除失败');
      _loadSchedules();
    });
  });
}

async function _saveScheduleSettings() {
  const num = id => {
    const v = document.getElementById(id).value.trim();
    return v === '' ? null : Number(v);
  };
  const msg = document.getElementById('sched-set-msg');
  const r = await _scheduleFetch('/api/schedules/settings', 'PATCH', {
    model_tools_enabled: document.getElementById('sched-set-model').checked,
    min_interval_minutes: num('sched-set-interval'),
    max_fires_per_day: num('sched-set-fires'),
    max_schedules: num('sched-set-max'),
    max_per_skill: num('sched-set-skill'),
    default_tz: document.getElementById('sched-set-tz').value.trim() || null,
  });
  if (r.ok) {
    _scheduleSettings = r.settings || _scheduleSettings;
    msg.className = 'schedule-msg';
    msg.textContent = '已保存';
  } else {
    msg.className = 'schedule-msg schedule-error';
    msg.textContent = r.error || '保存失败';
  }
}

function _openScheduleForm(s) {
  const slot = document.getElementById('schedule-form-slot');
  if (!slot) return;
  const isEdit = !!s;
  const v = s || { name: '', kind: 'cron', cron: '', at: '', tz: _scheduleSettings.default_tz || 'Asia/Shanghai',
                   text: '', skill: '', enabled: true };
  const atLocal = (v.at || '').replace(' ', 'T');
  slot.innerHTML = `
    <div class="task-card task-edit-form schedule-form">
      <label>名字</label>
      <input type="text" class="task-input" id="sf-name" value="${_escape(v.name)}">
      <label>类型</label>
      <select class="task-input" id="sf-kind">
        <option value="cron" ${v.kind !== 'once' ? 'selected' : ''}>重复</option>
        <option value="once" ${v.kind === 'once' ? 'selected' : ''}>只一次</option>
      </select>
      <div id="sf-cron-row">
        <label>Cron</label>
        <input type="text" class="task-input" id="sf-cron" value="${_escape(v.cron || '')}" placeholder="分 时 日 月 周，例如 0 8 * * *">
        <div class="schedule-preview" id="sf-preview"></div>
      </div>
      <div id="sf-at-row">
        <label>时间</label>
        <input type="datetime-local" class="task-input" id="sf-at" value="${_escape(atLocal)}">
      </div>
      <label>时区</label>
      <input type="text" class="task-input" id="sf-tz" value="${_escape(v.tz || '')}">
      <label>触发文字（到点推给机器人）</label>
      <textarea class="task-input" id="sf-text" rows="2">${_escape(v.text || '')}</textarea>
      <label>技能 slug（可空）</label>
      <input type="text" class="task-input" id="sf-skill" value="${_escape(v.skill || '')}">
      <label class="schedule-check"><input type="checkbox" id="sf-enabled" ${v.enabled ? 'checked' : ''}> 启用</label>
      <div class="schedule-msg schedule-error" id="sf-error"></div>
      <div class="task-actions" style="margin-top:10px">
        <button id="sf-save">保存</button>
        <button id="sf-cancel">取消</button>
      </div>
    </div>`;

  const kindSel = document.getElementById('sf-kind');
  const showKind = () => {
    const once = kindSel.value === 'once';
    document.getElementById('sf-cron-row').classList.toggle('hidden', once);
    document.getElementById('sf-at-row').classList.toggle('hidden', !once);
  };
  kindSel.addEventListener('change', showKind);
  showKind();

  let previewTimer = null;
  const preview = async () => {
    const out = document.getElementById('sf-preview');
    const cron = document.getElementById('sf-cron').value.trim();
    if (!cron) { out.textContent = ''; return; }
    try {
      const r = await _scheduleFetch('/api/schedules/preview', 'POST',
        { cron, tz: document.getElementById('sf-tz').value.trim() || null });
      out.className = r.ok ? 'schedule-preview' : 'schedule-preview schedule-error';
      out.textContent = r.ok ? `接下来：${(r.runs || []).join('、')}` : (r.error || '');
    } catch { out.textContent = ''; }
  };
  ['sf-cron', 'sf-tz'].forEach(id => document.getElementById(id).addEventListener('input', () => {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(preview, 400);
  }));
  preview();

  document.getElementById('sf-cancel').addEventListener('click', () => { slot.innerHTML = ''; });
  document.getElementById('sf-save').addEventListener('click', async () => {
    const kind = kindSel.value;
    const body = {
      name: document.getElementById('sf-name').value.trim(),
      kind,
      cron: kind === 'cron' ? document.getElementById('sf-cron').value.trim() : '',
      at: kind === 'once' ? document.getElementById('sf-at').value.replace('T', ' ') : '',
      tz: document.getElementById('sf-tz').value.trim(),
      text: document.getElementById('sf-text').value.trim(),
      skill: document.getElementById('sf-skill').value.trim(),
      enabled: document.getElementById('sf-enabled').checked,
    };
    const r = isEdit
      ? await _scheduleFetch(`/api/schedules/${encodeURIComponent(s.id)}`, 'PATCH', body)
      : await _scheduleFetch('/api/schedules', 'POST', body);
    if (!r.ok) {
      document.getElementById('sf-error').textContent = r.error || '保存失败';
      return;
    }
    slot.innerHTML = '';
    _loadSchedules();
  });
  slot.scrollIntoView({ block: 'nearest' });
}
