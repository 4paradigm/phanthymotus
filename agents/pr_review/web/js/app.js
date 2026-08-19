/** app.js — entry point: tab routing, polling loops, log tailing. */

import * as api from './api.js';
import * as views from './views.js';

const STATUS_POLL_MS = 5000;
const LOG_POLL_MS = 2000;
const PAGE_SIZE = 50;
// Bytes retained per log pane. Docker build logs run to megabytes; keeping all
// of it in the DOM makes the page crawl.
const LOG_PANE_MAX_CHARS = 400_000;

const TERMINAL = new Set([
  'review_done', 'build_success', 'build_failed', 'timeout', 'error', 'cancelled',
]);

const el = (id) => document.getElementById(id);

const state = {
  tab: 'overview',
  offset: 0,
  filterStatus: '',
  filterRepo: '',
  total: 0,
  jobId: null,
  knownRepos: [],
};

let statusTimer = null;
let logTimer = null;
// data-idx -> {offset, done}
const logCursors = new Map();
// The review trace has its own cursor, and the events are kept so the timeline
// can be rebuilt after renderDetail() replaces the DOM on the running->terminal
// transition. Without the copy, everything shown so far would vanish exactly
// when the review finishes.
const traceState = { jobId: null, offset: 0, events: [] };

// ── Bootstrap ────────────────────────────────────────────────────────────────

function init() {
  el('tab-bar').addEventListener('click', (e) => {
    const btn = e.target.closest('.tab-btn');
    if (btn) showTab(btn.dataset.tab);
  });

  el('btn-refresh').addEventListener('click', () => {
    if (state.tab === 'history') loadHistory();
    else if (state.tab === 'detail') loadDetail(state.jobId);
    else pollStatus();
  });

  el('btn-back').addEventListener('click', () => showTab('history'));

  el('filter-status').addEventListener('change', (e) => {
    state.filterStatus = e.target.value;
    state.offset = 0;
    loadHistory();
  });
  el('filter-repo').addEventListener('change', (e) => {
    state.filterRepo = e.target.value;
    state.offset = 0;
    loadHistory();
  });

  // Row clicks open a job, from either the overview or the history table.
  document.addEventListener('click', (e) => {
    const row = e.target.closest('tr[data-job]');
    if (row) { openJob(row.dataset.job); return; }

    const copy = e.target.closest('.btn-copy');
    if (copy) { copyText(copy); return; }

    const jump = e.target.closest('[data-log-bottom]');
    if (jump) {
      const pane = document.querySelector(
        `.log-pane[data-idx="${jump.dataset.logBottom}"]`);
      if (pane) pane.scrollTop = pane.scrollHeight;
    }
  });

  el('pager').addEventListener('click', (e) => {
    if (e.target.id === 'pg-prev') {
      state.offset = Math.max(0, state.offset - PAGE_SIZE);
      loadHistory();
    } else if (e.target.id === 'pg-next') {
      state.offset += PAGE_SIZE;
      loadHistory();
    }
  });

  // Deep-link support: #job/<id> opens a job directly, so a detail view can be
  // shared or reloaded.
  window.addEventListener('hashchange', applyHash);
  applyHash();

  startStatusPolling();
}

function applyHash() {
  const m = location.hash.match(/^#job\/(.+)$/);
  if (m) openJob(decodeURIComponent(m[1]), { skipHash: true });
  else if (location.hash === '#history') showTab('history');
  else showTab('overview');
}

// ── Tabs ─────────────────────────────────────────────────────────────────────

function showTab(tab) {
  state.tab = tab;

  document.querySelectorAll('.tab-btn').forEach((b) => {
    b.classList.toggle('active', b.dataset.tab === tab);
  });
  document.querySelectorAll('.tab-panel').forEach((p) => {
    p.classList.toggle('active', p.id === `panel-${tab}`);
  });

  // Only the visible view polls.
  stopLogPolling();
  if (tab === 'overview') {
    startStatusPolling();
    if (location.hash) location.hash = '';
  } else {
    stopStatusPolling();
  }

  if (tab === 'history') {
    if (location.hash !== '#history') location.hash = '#history';
    loadHistory();
  }
}

// ── Overview ─────────────────────────────────────────────────────────────────

function startStatusPolling() {
  stopStatusPolling();
  pollStatus();
  statusTimer = setInterval(pollStatus, STATUS_POLL_MS);
}

function stopStatusPolling() {
  if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
}

async function pollStatus() {
  try {
    const s = await api.getStatus();
    setConn('ok', 'connected');
    views.renderStats(el('stat-grid'), s);
    views.renderActive(el('active-body'), el('active-meta'), s);
    views.renderPoller(el('poller-body'), s.poller);
    views.renderConfig(el('config-body'), s.config);
    syncRepoFilter(s.config?.repos || s.history?.repos || []);
  } catch (err) {
    setConn('err', 'unreachable');
  }
}

function setConn(cls, text) {
  const dot = el('conn-dot');
  dot.className = `conn-dot ${cls}`;
  el('conn-text').textContent = text;
}

function syncRepoFilter(repos) {
  const same = repos.length === state.knownRepos.length &&
    repos.every((r, i) => r === state.knownRepos[i]);
  if (same) return;
  state.knownRepos = repos.slice();

  const sel = el('filter-repo');
  const current = sel.value;
  sel.innerHTML = '<option value="">All</option>' +
    repos.map((r) => `<option value="${api.esc(r)}">${api.esc(r)}</option>`).join('');
  if (repos.includes(current)) sel.value = current;
}

// ── History ──────────────────────────────────────────────────────────────────

async function loadHistory() {
  try {
    const res = await api.getJobs({
      limit: PAGE_SIZE,
      offset: state.offset,
      status: state.filterStatus,
      repo: state.filterRepo,
    });
    state.total = res.total;
    views.renderHistory(el('history-body'), res.jobs);
    views.renderPager(el('pager'), res);
    el('history-meta').textContent = `${res.total} job(s)`;
    setConn('ok', 'connected');
  } catch (err) {
    el('history-body').innerHTML =
      `<div class="banner err">Failed to load history: ${api.esc(err.message)}</div>`;
    setConn('err', 'unreachable');
  }
}

// ── Job detail ───────────────────────────────────────────────────────────────

function openJob(id, { skipHash = false } = {}) {
  state.jobId = id;
  if (!skipHash) location.hash = `#job/${encodeURIComponent(id)}`;
  showTab('detail');
  loadDetail(id);
}

async function loadDetail(id) {
  if (!id) return;
  const body = el('detail-body');
  try {
    const job = await api.getJob(id);
    views.renderDetail(body, job);
    setConn('ok', 'connected');

    logCursors.clear();
    (job.build_results || []).forEach((b) => {
      logCursors.set(String(b.idx), { offset: 0, done: false });
    });

    if (traceState.jobId !== id) {
      traceState.jobId = id;
      traceState.offset = 0;
      traceState.events = [];
    } else {
      // Same job, re-rendered: replay what we already have so the timeline is
      // not lost, then continue from the existing cursor.
      drawTrace();
    }

    await tickLogs();
    await tickTrace(id);
    // Keep tailing while the job can still produce output. A finished job's
    // logs are static, so one read is enough.
    if (!TERMINAL.has(job.status)) startLogPolling(id);
  } catch (err) {
    body.innerHTML =
      `<div class="banner err">Failed to load job: ${api.esc(err.message)}</div>`;
    setConn('err', 'unreachable');
  }
}

function startLogPolling(jobId) {
  stopLogPolling();
  logTimer = setInterval(async () => {
    await tickLogs();
    await tickTrace(jobId);
    // Poll the job itself too, so the page reflects the transition out of
    // running and then stops tailing.
    try {
      const job = await api.getJob(jobId);
      if (TERMINAL.has(job.status)) {
        stopLogPolling();
        views.renderDetail(el('detail-body'), job);
        drawTrace();
        await tickLogs();
        await tickTrace(jobId);
      }
    } catch { /* transient — the next tick retries */ }
  }, LOG_POLL_MS);
}

function stopLogPolling() {
  if (logTimer) { clearInterval(logTimer); logTimer = null; }
}

/** Draw every accumulated trace event into the (empty) timeline container. */
function drawTrace() {
  const box = document.querySelector('.trace[data-trace-job]');
  if (!box || !traceState.events.length) return;
  box.textContent = '';
  views.appendTraceEvents(box, traceState.events);
}

/** Fetch new review-trace events and append them to the timeline. */
async function tickTrace(jobId) {
  const box = document.querySelector('.trace[data-trace-job]');
  if (!box) return;
  try {
    const res = await api.getReviewTrace(jobId, traceState.offset);
    traceState.offset = res.offset;
    if (res.events?.length) {
      traceState.events.push(...res.events);
      views.appendTraceEvents(box, res.events);
      const meta = document.querySelector('[data-trace-meta]');
      const rounds = traceState.events.filter((e) => e.kind === 'round').length;
      const tools = traceState.events.filter((e) => e.kind === 'tool').length;
      if (meta && rounds) {
        meta.textContent = `${rounds} round${rounds === 1 ? '' : 's'} · ` +
          `${tools} tool call${tools === 1 ? '' : 's'}`;
      }
    }
  } catch {
    // 404 until the review starts — the same "not there yet" case as a build
    // log for a queued build. Stay quiet.
  }
}

/** Append any new bytes to each visible log pane. */
async function tickLogs() {
  const panes = document.querySelectorAll('.log-pane[data-job]');
  for (const pane of panes) {
    const idx = pane.dataset.idx;
    const cursor = logCursors.get(idx) || { offset: 0, done: false };
    try {
      const res = await api.getLog(pane.dataset.job, idx, cursor.offset);
      cursor.offset = res.offset;
      logCursors.set(idx, cursor);
      if (res.content) appendLog(pane, res.content);
      setLogState(idx, res);
    } catch {
      // A log can legitimately not exist yet (build queued) — stay quiet.
    }
  }
}

/**
 * Append text to a log pane.
 *
 * Two behaviours borrowed from agent-core's activity-log: check whether the
 * user is at the bottom *before* appending and only autoscroll if so (otherwise
 * tailing yanks the view away from someone reading an error further up), and cap
 * retained text so a multi-megabyte build log does not bloat the DOM.
 *
 * textContent, never innerHTML — build output is attacker-influenced.
 */
function appendLog(pane, text) {
  const atBottom = pane.scrollHeight - pane.scrollTop - pane.clientHeight < 40;

  pane.textContent += text;
  if (pane.textContent.length > LOG_PANE_MAX_CHARS) {
    const trimmed = pane.textContent.slice(-LOG_PANE_MAX_CHARS);
    // Drop the leading partial line so the pane never opens mid-token.
    const nl = trimmed.indexOf('\n');
    pane.textContent = '… (earlier output trimmed)\n' +
      (nl >= 0 ? trimmed.slice(nl + 1) : trimmed);
  }

  if (atBottom) pane.scrollTop = pane.scrollHeight;
}

function setLogState(idx, res) {
  const node = document.querySelector(`[data-log-state="${idx}"]`);
  if (!node) return;
  const kb = (res.size / 1024).toFixed(1);
  node.textContent = logTimer ? `${kb} KB · tailing` : `${kb} KB`;
}

// ── Clipboard ────────────────────────────────────────────────────────────────

async function copyText(btn) {
  const value = btn.dataset.copy || '';
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    // clipboard API needs a secure context; over a plain-HTTP tunnel it may be
    // unavailable, so fall back to a hidden textarea.
    const ta = document.createElement('textarea');
    ta.value = value;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch { /* give up quietly */ }
    document.body.removeChild(ta);
  }
  const original = btn.textContent;
  btn.textContent = 'copied';
  btn.classList.add('done');
  setTimeout(() => {
    btn.textContent = original;
    btn.classList.remove('done');
  }, 1200);
}

init();
