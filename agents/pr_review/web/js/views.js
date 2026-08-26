/** views.js — renderers for the overview, history, and job detail panels. */

import {
  esc, renderMarkdown, fmtDuration, fmtTime, fmtRelative, fmtRelativeIso,
  shortSha, shortRepo, targetLabel, prUrl,
} from './api.js';

// Statuses from which a job will not advance — mirrors the server's set.
const TERMINAL = new Set([
  'review_done', 'build_success', 'build_failed', 'timeout', 'error', 'cancelled',
]);

// ── Shared cells ─────────────────────────────────────────────────────────────

/**
 * The commit cell.
 *
 * Shows the *build ref* — the worktree HEAD after the PR was merged onto base,
 * which is the sha the build scripts shorten into `release.YYMMDD.<7hex>`. That
 * makes the column the thing you can actually search a registry for. The PR head
 * sha is not: it never names an image. Falls back to it for jobs recorded before
 * the build ref was captured, and the tooltip carries all three ids in full.
 */
function commitCell(j) {
  const shown = j.build_ref_sha || j.head_sha;
  const lines = [
    `PR head:      ${j.head_sha || '—'}`,
    `build ref:    ${j.build_ref_sha || '—'}   (names the image tag)`,
    `merge commit: ${j.merge_commit_sha || 'not merged yet'}`,
  ].join('\n');
  return `<td class="mono" title="${esc(lines)}">${esc(shortSha(shown))}</td>`;
}

/**
 * Who the work belongs to: the PR's author, not whoever typed the trigger.
 *
 * Falls back to `requester` for rows written before the author was recorded —
 * usually the same person, and better than a blank column.
 */
function byCell(j) {
  return `<td>${esc(j.pr_author || j.requester || '—')}</td>`;
}

// ── Overview ─────────────────────────────────────────────────────────────────

export function renderStats(el, s) {
  const hist = s.history || {};
  const tiles = [
    { label: 'Queued', value: s.queue_depth, cls: s.queue_depth > 0 ? 'yellow' : '' },
    { label: 'In flight', value: s.active_jobs, cls: s.active_jobs > 0 ? 'blue' : '' },
    { label: 'Processed', value: s.total_processed, cls: 'green' },
    { label: 'History', value: hist.total ?? 0, cls: '' },
    {
      label: 'Workers',
      value: `${s.active_jobs}/${s.config?.max_concurrent_jobs ?? '?'}`,
      cls: '',
    },
  ];
  el.innerHTML = tiles.map((t) => `
    <div class="stat-tile ${t.cls}">
      <div class="stat-value">${esc(t.value)}</div>
      <div class="stat-label">${esc(t.label)}</div>
    </div>
  `).join('');
}

export function renderActive(bodyEl, metaEl, s) {
  const active = s.active || [];
  metaEl.textContent = active.length ? `${active.length} running` : '';

  if (!active.length) {
    bodyEl.innerHTML = emptyState('◎', 'Nothing in flight',
      'Comment /request_bot_review on a PR to trigger a review.');
    return;
  }

  bodyEl.innerHTML = `
    <table class="tbl">
      <thead><tr>
        <th>PR</th><th>Repo</th><th>Commit</th><th>Stage</th>
        <th>Attempt</th><th>By</th><th class="num">Elapsed</th>
      </tr></thead>
      <tbody>
        ${active.map((j) => `
          <tr class="clickable" data-job="${esc(j.id)}">
            <td class="mono">#${esc(j.pr_number)}</td>
            <td>${esc(shortRepo(j.repo))}</td>
            ${commitCell(j)}
            <td>${stageCell(j)}</td>
            <td class="mono">${esc(j.attempt)}</td>
            ${byCell(j)}
            <td class="num">${esc(fmtDuration(j.elapsed))}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
}

/**
 * Render the current pipeline stage.
 *
 * `status` alone sits at "running" through a fetch, a merge, several builds and
 * the review, which reads as a hang. The stage plus how long it has been in that
 * stage is what distinguishes slow from stuck.
 */
export function stageCell(j) {
  if (!j.stage || j.stage === 'done') {
    return `<span class="pill ${esc(j.status)}">${esc(j.status)}</span>`;
  }
  const detail = j.stage_detail ? ` <span class="stage-detail">${esc(j.stage_detail)}</span>` : '';
  const held = j.stage_elapsed != null && j.stage_elapsed >= 20
    ? ` <span class="stage-held">${esc(fmtDuration(j.stage_elapsed))}</span>`
    : '';
  return `<span class="pill running">${esc(j.stage)}</span>${detail}${held}`;
}

export function renderPoller(el, p) {
  if (!p || p.enabled === false) {
    el.innerHTML = `<dl class="kv"><dt>Mode</dt><dd class="plain">Disabled — webhook only</dd></dl>`;
    return;
  }
  const stale = _pollIsStale(p);
  el.innerHTML = `
    <dl class="kv">
      <dt>Interval</dt><dd>${esc(p.interval_seconds)}s</dd>
      <dt>Last poll</dt>
      <dd>${esc(fmtRelativeIso(p.last_poll_at))}
        ${stale ? '<span class="pill timeout">stale</span>' : ''}</dd>
      <dt>Cycles</dt><dd>${esc(p.poll_count ?? 0)}</dd>
      <dt>Triggers seen</dt><dd>${esc(p.triggers_found ?? 0)}</dd>
      <dt>Last error</dt>
      <dd>${p.last_error
        ? `<span style="color:var(--red)">${esc(p.last_error)}</span>`
        : '<span style="color:var(--text-dim)">none</span>'}</dd>
    </dl>`;
}

export function renderConfig(el, c) {
  if (!c) { el.innerHTML = ''; return; }
  el.innerHTML = `
    <dl class="kv">
      <dt>Repos</dt><dd>${(c.repos || []).map((r) => esc(r)).join('<br>') || '—'}</dd>
      <dt>Concurrency</dt><dd>${esc(c.max_concurrent_jobs)}</dd>
      <dt title="No output for this long means a build is stuck, and it is killed. This is the bound that normally fires.">Build idle timeout</dt>
      <dd>${esc(fmtDuration(c.build_idle_timeout_seconds))}</dd>
      <dt title="Absolute backstop for one build, for a build that prints forever without finishing.">Build timeout</dt>
      <dd>${esc(fmtDuration(c.build_timeout_seconds))}</dd>
      <dt>Job timeout</dt><dd>${esc(fmtDuration(c.job_timeout_seconds))}</dd>
      <dt>Max attempts</dt><dd>${esc(c.max_attempts)}</dd>
      <dt>Mirror</dt><dd>${esc(c.mirror)}</dd>
      <dt>Retention</dt><dd>${esc(c.job_history_days)}d</dd>
      <dt>Webhook</dt>
      <dd><span class="pill ${c.webhook_enabled ? 'ok' : 'cancelled'}">${
        c.webhook_enabled ? 'enabled' : 'disabled'}</span></dd>
      <dt>LLM review</dt>
      <dd>${c.llm_configured
        ? `<span class="pill ok">on</span> ${esc(c.llm_model)}`
        : '<span class="pill cancelled">not configured</span>'}</dd>
    </dl>`;
}

/** A poller that has not run in >3 intervals is not keeping up. */
function _pollIsStale(p) {
  if (!p.last_poll_at) return true;
  const t = Date.parse(p.last_poll_at);
  if (Number.isNaN(t)) return true;
  return (Date.now() - t) / 1000 > Math.max(90, (p.interval_seconds || 30) * 3);
}

// ── History ──────────────────────────────────────────────────────────────────

export function renderHistory(el, jobs) {
  if (!jobs.length) {
    el.innerHTML = emptyState('◈', 'No reviews yet',
      'Reviews appear here once /request_bot_review is used on a PR.');
    return;
  }

  el.innerHTML = `
    <table class="tbl">
      <thead><tr>
        <th>PR</th><th>Repo</th><th>Commit</th><th>Status</th>
        <th>Builds</th><th>By</th><th>Via</th>
        <th class="num">Elapsed</th><th class="num">When</th>
      </tr></thead>
      <tbody>
        ${jobs.map((j) => `
          <tr class="clickable" data-job="${esc(j.id)}">
            <td class="mono">#${esc(j.pr_number)}</td>
            <td>${esc(shortRepo(j.repo))}</td>
            ${commitCell(j)}
            <td>
              ${TERMINAL.has(j.status)
                ? `<span class="pill ${esc(j.status)}">${esc(j.status)}</span>`
                : stageCell(j)}
              ${j.attempt > 1 ? `<span class="pill">try ${esc(j.attempt)}</span>` : ''}
            </td>
            <td>${_buildSummary(j.build_results)}</td>
            ${byCell(j)}
            <td>${esc(j.source)}</td>
            <td class="num">${esc(fmtDuration(j.elapsed))}</td>
            <td class="num" title="${esc(fmtTime(j.created_at))}">${esc(fmtRelative(j.created_at))}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
}

/**
 * A build's outcome pill.
 *
 * `success` is tri-state: null means the row is the placeholder the worker writes
 * before a build starts, so the dashboard has a log pane to tail. Rendering that
 * as "failed" showed a FAILED build beside a job that was building normally.
 */
function buildPill(b) {
  if (b.success === null || b.success === undefined) return 'running';
  return b.success ? 'ok' : 'fail';
}

function buildLabel(b) {
  if (b.success === null || b.success === undefined) return 'building';
  if (b.success) return 'success';
  // A build the agent killed is a different fact from one that failed to
  // compile, and the fix is different too — so it does not read as "failed".
  if (b.timeout_kind === 'idle') return 'stalled';
  if (b.timeout_kind === 'cap') return 'timed out';
  return 'failed';
}

function _buildSummary(results) {
  if (!results || !results.length) return '<span style="color:var(--text-dim)">—</span>';
  return results.map((b) =>
    `<span class="pill ${buildPill(b)}">${esc(targetLabel(b))}</span>`
  ).join(' ');
}

export function renderPager(el, { total, limit, offset }) {
  const page = Math.floor(offset / limit) + 1;
  const pages = Math.max(1, Math.ceil(total / limit));
  el.innerHTML = `
    <button class="btn-ghost btn-sm" id="pg-prev" ${offset <= 0 ? 'disabled' : ''}>Previous</button>
    <span class="pager-info">Page ${page} / ${pages} · ${total} total</span>
    <button class="btn-ghost btn-sm" id="pg-next" ${offset + limit >= total ? 'disabled' : ''}>Next</button>`;
}

// ── Job detail ───────────────────────────────────────────────────────────────

export function renderDetail(el, job) {
  el.innerHTML = [
    _detailMeta(job),
    _detailBuilds(job),
    _detailPRContext(job),
    _detailReviewProcess(job),
    _detailReview(job),
    _detailChangeAudit(job),
    _detailFindings(job),
    _detailErrors(job),
  ].join('');
}

function _detailMeta(j) {
  const o = j.options || {};
  const mode = o.skip_build ? 'review only'
    : o.build_only ? 'build only'
    : 'build + review';
  const running = !TERMINAL.has(j.status);
  return `
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">
          <a href="${esc(prUrl(j.repo, j.pr_number))}" target="_blank" rel="noopener"
             style="color:var(--accent);text-decoration:none">
            ${esc(shortRepo(j.repo))} #${esc(j.pr_number)}
          </a>
        </h2>
        <span class="pill ${esc(j.status)}">${esc(j.status)}</span>
        <span class="card-meta">${esc(j.id)}</span>
      </div>
      <div class="card-body">
        ${running ? `<div class="stage-banner">${stageCell(j)}</div>` : ''}
        <dl class="kv">
          ${j.pr_title ? `<dt>Title</dt><dd class="plain">${esc(j.pr_title)}</dd>` : ''}
          <dt>PR head</dt><dd>${esc(j.head_sha || '—')}</dd>
          <dt title="The sha the build scripts turn into release.YYMMDD.&lt;7hex&gt;">Build ref</dt>
          <dd>${j.build_ref_sha
            ? `${esc(j.build_ref_sha)} <span class="stage-detail">names the image tag</span>`
            : '—'}</dd>
          <dt>Merge commit</dt>
          <dd>${j.merge_commit_sha
            ? esc(j.merge_commit_sha)
            : '<span style="color:var(--text-dim)">not merged yet</span>'}</dd>
          <dt>Branch</dt><dd>${esc(j.head_ref || '—')} → ${esc(j.base_ref || '—')}</dd>
          <dt>PR author</dt><dd class="plain">${esc(j.pr_author || j.requester || '—')}</dd>
          <dt>Triggered via</dt><dd class="plain">${esc(j.source || '—')}</dd>
          <dt>Mode</dt><dd class="plain">${esc(mode)}${
            (o.force_targets || []).length
              ? ` · forced: ${esc((o.force_targets || []).join(', '))}` : ''}${
            (o.perception_variants || []).length
              ? ` · JetPack: ${esc((o.perception_variants || []).join(', '))}` : ''}</dd>
          <dt>Attempt</dt><dd>${esc(j.attempt)}</dd>
          <dt>Created</dt><dd>${esc(fmtTime(j.created_at))}</dd>
          <dt>Started</dt><dd>${esc(fmtTime(j.started_at))}</dd>
          <dt>Finished</dt><dd>${esc(fmtTime(j.finished_at))}</dd>
          <dt>Duration</dt><dd>${esc(fmtDuration(j.elapsed))}</dd>
        </dl>
      </div>
    </div>`;
}

function _detailBuilds(j) {
  const results = j.build_results || [];
  if (!results.length) {
    return `<div class="card">
      <div class="card-header"><h2 class="card-title">Builds</h2></div>
      <div class="card-body">${emptyState('◇', 'No builds',
        'The changes did not touch a buildable component, or build was skipped.')}</div>
    </div>`;
  }

  const rows = results.map((b) => `
    <tr>
      <td>${esc(targetLabel(b))}</td>
      <td><span class="pill ${buildPill(b)}">${esc(buildLabel(b))}</span></td>
      <td>${b.image_tag ? `
        <div class="copy-cell">
          <span class="mono">${esc(b.image_tag)}</span>
          <button class="btn-copy" data-copy="${esc(b.image_tag)}">copy</button>
        </div>` : '<span style="color:var(--text-dim)">—</span>'}</td>
      <td>${esc(fmtDuration(b.duration_seconds))}</td>
    </tr>`).join('');

  // Each build gets its own log pane; app.js attaches the tailing loop by
  // reading data-job / data-idx off the pane.
  const logs = results.map((b) => `
    <div class="log-wrap" data-log-block="${esc(b.idx)}">
      <div class="log-toolbar">
        <strong>${esc(targetLabel(b))}</strong>
        <span class="log-toolbar-spacer"></span>
        <span class="log-tail-state" data-log-state="${esc(b.idx)}"></span>
        <button class="btn-ghost btn-sm" data-log-bottom="${esc(b.idx)}">Jump to end</button>
      </div>
      <pre class="log-pane" data-job="${esc(j.id)}" data-idx="${esc(b.idx)}"></pre>
    </div>`).join('');

  return `
    <div class="card">
      <div class="card-header"><h2 class="card-title">Builds</h2></div>
      <div class="card-body no-pad">
        <table class="tbl">
          <thead><tr><th>Target</th><th>Status</th><th>Image</th><th>Took</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      ${logs}
    </div>`;
}

function _detailReview(j) {
  if (!j.review_text) return '';
  const r = j.review || {};
  // A review cut short must not look like a review that found nothing, so the
  // stop reason is shown next to the text rather than only in the PR comment.
  const meta = r.rounds
    ? `<span class="card-meta">${r.rounds} rounds · ${r.tool_calls || 0} tool calls</span>`
    : '';
  const CUT = {
    max_rounds: 'Cut short — round limit reached; may be incomplete.',
    timeout: 'Cut short — time limit reached; may be incomplete.',
    error: 'Ended on an error; may be incomplete.',
  };
  const warn = CUT[r.stopped_reason]
    ? `<div class="finding"><span class="pill sev-warning">warning</span>
         <div class="finding-msg">${esc(CUT[r.stopped_reason])}</div></div>`
    : '';
  return `
    <div class="card">
      <div class="card-header"><h2 class="card-title">Code review</h2>${meta}</div>
      <div class="card-body">
        ${warn}
        <div class="review-body">${renderMarkdown(j.review_text)}</div>
      </div>
    </div>`;
}

function _detailChangeAudit(j) {
  const big = j.large_files || [];
  const infra = j.infra_files || [];
  if (!big.length && !infra.length) return '';
  const shared = new Set(j.shared_base_files || []);

  const bigRows = big.map((e) => `
    <div class="finding">
      <span class="pill sev-error">${Math.round(e.bytes / 1024)}KB</span>
      <div class="finding-file">${esc(e.file)}</div>
    </div>`).join('');
  const infraRows = infra.map((f) => `
    <div class="finding">
      <span class="pill sev-${shared.has(f) ? 'error' : 'warning'}">
        ${shared.has(f) ? 'shared' : 'infra'}</span>
      <div>
        <div class="finding-file">${esc(f)}</div>
        ${shared.has(f)
          ? '<div class="finding-msg">Affects every component built on it, across both repositories.</div>'
          : ''}
      </div>
    </div>`).join('');

  return `
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Change audit</h2>
        <span class="card-meta">${big.length} large · ${infra.length} infra</span>
      </div>
      <div class="card-body">
        ${big.length ? `<div class="finding-msg">Files over the size limit — these
          belong in COS, fetched at build time.</div>${bigRows}` : ''}
        ${infraRows}
      </div>
    </div>`;
}

/**
 * What the PR said about itself — the context the reviewer was given.
 *
 * Shown so a reader can judge the review against the same information it had:
 * a review that contradicts the author's claims is doing its job, and that is
 * only visible with both side by side.
 *
 * The description is PR-authored, so it renders through renderMarkdown(), which
 * escapes before substituting.
 */
function _detailPRContext(j) {
  const body = (j.pr_body || '').trim();
  const ctx = j.pr_context || {};
  if (!body && !ctx.description_missing) return '';

  const meta = [
    body ? `${body.length} chars` : '',
    ctx.comments_used ? `${ctx.comments_used} comments fed in` : '',
    ctx.comments_dropped ? `${ctx.comments_dropped} omitted` : '',
  ].filter(Boolean).join(' · ');

  const warn = ctx.description_missing
    ? `<div class="finding">
         <span class="pill sev-warning">warning</span>
         <div class="finding-msg">No usable description — empty or an unfilled
           template. The reviewer was told to raise this.</div>
       </div>`
    : '';

  return `
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">PR description</h2>
        <span class="card-meta">${esc(meta)}</span>
      </div>
      <div class="card-body">
        ${warn}
        ${body ? `<div class="review-body">${renderMarkdown(body)}</div>` : ''}
      </div>
    </div>`;
}

function _detailReviewProcess(j) {
  // Rendered while the review is still running too, so the card exists before
  // there is any review text to show. Events arrive via appendTraceEvents().
  const running = j.stage === 'generating review' || j.stage === 'running rule checks';
  if (!j.has_review_trace && !running) return '';
  const r = j.review || {};
  const tools = r.tool_calls || 0;
  const meta = r.rounds
    ? `${r.rounds} round${r.rounds === 1 ? '' : 's'} · ` +
      `${tools} tool call${tools === 1 ? '' : 's'}`
    : 'running…';
  return `
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Review process</h2>
        <span class="card-meta" data-trace-meta>${esc(meta)}</span>
      </div>
      <div class="card-body no-pad">
        <div class="trace" data-trace-job="${esc(j.id)}"></div>
      </div>
    </div>`;
}

/**
 * Append trace events to the process timeline.
 *
 * Incremental on purpose: re-rendering the whole timeline each poll would
 * collapse any <details> the reader has open, which is exactly the pane they are
 * reading. So rounds are found-or-created and rows appended.
 *
 * Everything here is built with createElement/textContent rather than an HTML
 * string. Tool results are file contents from an untrusted PR — this is the one
 * place they are shown verbatim, so there is no interpolation to get wrong.
 */
export function appendTraceEvents(container, events) {
  for (const ev of events) {
    switch (ev.kind) {
      case 'setup':   container.appendChild(_traceSetup(ev)); break;
      case 'round':   container.appendChild(_traceRound(ev)); break;
      case 'tool':    _traceRoundBody(container, ev.round)
                        .appendChild(ev.markdown ? _traceReview(ev) : _traceTool(ev));
                      break;
      case 'nudge':   _traceRoundBody(container, ev.round)
                        .appendChild(_traceNote('warning',
                          `Returned nothing (${ev.attempt}/${ev.limit}) — ${ev.reason || 'retrying'}`)); break;
      // A retried round is otherwise indistinguishable from a slow model, which
      // sends you debugging the wrong thing.
      case 'llm_retry':
                      _traceRoundBody(container, ev.round)
                        .appendChild(_traceNote('warning',
                          `LLM call failed (retry ${ev.attempt}/${ev.limit}, ` +
                          `waiting ${ev.delay}s) — ${ev.error || 'transient error'}`)); break;
      case 'finish':  container.appendChild(_traceFinish(ev)); break;
      // 'refusal' duplicates what the tool row already shows as [refused].
      default: break;
    }
  }
}

function _traceBlock(summary, meta, open = false) {
  const d = document.createElement('details');
  d.className = 'trace-block';
  if (open) d.open = true;
  const s = document.createElement('summary');
  s.className = 'trace-summary';
  const title = document.createElement('span');
  title.className = 'trace-title';
  title.textContent = summary;
  s.appendChild(title);
  if (meta) {
    const m = document.createElement('span');
    m.className = 'trace-meta';
    m.textContent = meta;
    s.appendChild(m);
  }
  d.appendChild(s);
  const body = document.createElement('div');
  body.className = 'trace-body';
  d.appendChild(body);
  return d;
}

function _kv(body, label, value) {
  if (value === undefined || value === null || value === '' ||
      (Array.isArray(value) && !value.length)) return;
  const row = document.createElement('div');
  row.className = 'trace-kv';
  const k = document.createElement('span');
  k.className = 'trace-k';
  k.textContent = label;
  const v = document.createElement('span');
  v.className = 'trace-v';
  v.textContent = Array.isArray(value) ? value.join(', ') : String(value);
  row.append(k, v);
  body.appendChild(row);
}

function _traceSetup(ev) {
  // Open by default: what the reviewer was told is the context for everything
  // below it, and it is short.
  const block = _traceBlock(
    ev.attempt > 1 ? `Setup — review restarted (attempt ${ev.attempt})` : 'Setup',
    ev.model || '', true);
  const body = block.querySelector('.trace-body');
  _kv(body, 'component', ev.component);
  _kv(body, 'rules', `${(ev.rules || []).join(' + ')}  (${ev.rules_chars || 0} chars)`);
  _kv(body, 'docs to read', ev.docs);
  _kv(body, 'compare against', ev.references);
  _kv(body, 'budget', `${ev.max_rounds} rounds / ${ev.timeout_seconds}s`);
  _kv(body, 'changed files', ev.changed_files);
  _kv(body, 'large files', ev.large_files);
  _kv(body, 'infrastructure', ev.infra_files);
  _kv(body, 'shared base', ev.shared_base_files);
  return block;
}

function _traceRound(ev) {
  const bits = [];
  if (ev.elapsed !== undefined) bits.push(`${ev.elapsed}s`);
  if (ev.prompt_tokens) {
    // cached_tokens is the only visible signal that the stable-system-prompt
    // design is actually getting prefix cache hits.
    bits.push(ev.cached_tokens
      ? `${_n(ev.prompt_tokens)} prompt (${_n(ev.cached_tokens)} cached)`
      : `${_n(ev.prompt_tokens)} prompt`);
  }
  if (ev.completion_tokens) bits.push(`${_n(ev.completion_tokens)} out`);
  if (ev.reasoning_tokens) bits.push(`${_n(ev.reasoning_tokens)} reasoning`);

  const block = _traceBlock(`Round ${ev.round}`, bits.join(' · '), true);
  block.dataset.traceRound = String(ev.round);
  const body = block.querySelector('.trace-body');
  if (ev.error) body.appendChild(_traceNote('error', ev.error));
  if (ev.content) {
    // The model narrating alongside its tool calls — the closest thing to
    // visible reasoning this router returns.
    const p = document.createElement('div');
    p.className = 'trace-narration';
    p.textContent = ev.content;
    body.appendChild(p);
  } else if (ev.tools?.length) {
    // The header advertises N output tokens; without this there is nothing on
    // screen to account for them. Those tokens *were* the tool calls, whose
    // arguments each row shows under "called".
    const p = document.createElement('div');
    p.className = 'trace-narration muted';
    p.textContent =
      `No prose this round — the output was ${ev.tools.length} tool call` +
      `${ev.tools.length === 1 ? '' : 's'}: ${ev.tools.join(', ')}`;
    body.appendChild(p);
  }
  return block;
}

/** The body of round N, creating a placeholder block if it is not there yet. */
function _traceRoundBody(container, round) {
  const key = String(round ?? 0);
  // The *last* block with this number, not the first: one trace file can hold
  // more than one review of the same job — a restarted review, or a job-level
  // retry — and each starts counting rounds at 1 again. Matching the first
  // block would file the second review's round 1 under the first review's.
  const all = container.querySelectorAll(`[data-trace-round="${CSS.escape(key)}"]`);
  let block = all[all.length - 1];
  if (!block) {
    block = _traceBlock(`Round ${key}`, '', true);
    block.dataset.traceRound = key;
    container.appendChild(block);
  }
  return block.querySelector('.trace-body');
}

function _traceTool(ev) {
  const row = document.createElement('details');
  row.className = 'trace-tool';

  const s = document.createElement('summary');
  s.className = 'trace-tool-head';

  const name = document.createElement('span');
  name.className = 'trace-tool-name';
  name.textContent = ev.name || '?';

  const summary = document.createElement('span');
  summary.className = 'trace-tool-arg';
  summary.textContent = ev.summary || '';

  s.append(name, summary);

  if (ev.refused || ev.error) {
    const pill = document.createElement('span');
    pill.className = `pill ${ev.refused ? 'sev-warning' : 'sev-error'}`;
    pill.textContent = ev.refused ? 'refused' : 'error';
    s.appendChild(pill);
  }
  const size = document.createElement('span');
  size.className = 'trace-tool-size';
  size.textContent = [
    ev.bytes !== undefined ? _bytes(ev.bytes) : '',
    ev.ms ? `${ev.ms}ms` : '',
  ].filter(Boolean).join(' · ');
  s.appendChild(size);

  row.appendChild(s);

  // The literal call the model emitted. The summary line above is a readable
  // rendering of it; this is the arguments verbatim, which is what the round's
  // output tokens were actually spent on.
  if (ev.args && Object.keys(ev.args).length) {
    const call = document.createElement('pre');
    call.className = 'trace-call';
    call.textContent = `${ev.name}(${JSON.stringify(ev.args, null, 1)
      .replace(/\n\s*/g, ' ')})`;
    row.appendChild(call);
  }

  const pre = document.createElement('pre');
  pre.className = 'trace-result';
  pre.textContent = ev.result || '(no output)';
  row.appendChild(pre);
  return row;
}

/**
 * The written review, as its own block rather than a tool row.
 *
 * This is what the loop was for, so it renders as markdown like the review card
 * instead of being dumped into a <pre> the way tool output is. renderMarkdown()
 * escapes before applying its patterns, which is what makes that safe.
 */
function _traceReview(ev) {
  const block = _traceBlock('Review written', ev.summary || '', true);
  block.classList.add('trace-review');
  const body = block.querySelector('.trace-body');
  const md = document.createElement('div');
  md.className = 'review-body';
  md.innerHTML = renderMarkdown(ev.result || '');
  body.appendChild(md);
  return block;
}

function _traceNote(sev, text) {
  const d = document.createElement('div');
  d.className = 'trace-note';
  const pill = document.createElement('span');
  pill.className = `pill sev-${sev}`;
  pill.textContent = sev;
  const msg = document.createElement('span');
  msg.textContent = text;
  d.append(pill, msg);
  return d;
}

function _traceFinish(ev) {
  const REASON = {
    finished: 'Finished — review written',
    max_rounds: 'Stopped: round limit reached — the review may be incomplete',
    timeout: 'Stopped: time limit reached — the review may be incomplete',
    error: 'Stopped on an error — the review may be incomplete',
  };
  const d = document.createElement('div');
  d.className = `trace-finish ${ev.stopped_reason === 'finished' ? 'ok' : 'warn'}`;
  const label = document.createElement('span');
  label.textContent = REASON[ev.stopped_reason] || ev.stopped_reason || 'Finished';
  d.appendChild(label);
  const meta = document.createElement('span');
  meta.className = 'trace-meta';
  meta.textContent = [
    ev.rounds ? `${ev.rounds} rounds` : '',
    ev.tool_calls ? `${ev.tool_calls} tool calls` : '',
    ev.t !== undefined ? `${ev.t}s total` : '',
  ].filter(Boolean).join(' · ');
  d.appendChild(meta);
  if (ev.error) d.appendChild(_traceNote('error', ev.error));
  return d;
}

const _n = (v) => Number(v).toLocaleString('en-US');

function _bytes(n) {
  if (n < 1024) return `${n} B`;
  return `${(n / 1024).toFixed(1)} KB`;
}


function _detailFindings(j) {
  const findings = j.findings || [];
  if (!findings.length) return '';
  return `
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Rule checks</h2>
        <span class="card-meta">${findings.length}</span>
      </div>
      <div class="card-body">
        ${findings.map((f) => `
          <div class="finding">
            <span class="pill sev-${esc(f.severity)}">${esc(f.severity)}</span>
            <div>
              <div class="finding-file">${esc(f.file)}</div>
              <div class="finding-msg">${esc(f.message)}</div>
            </div>
          </div>`).join('')}
      </div>
    </div>`;
}

function _detailErrors(j) {
  const errs = j.attempt_errors || [];
  if (!errs.length && !j.error) return '';

  // Error text renders through esc() into <pre> rather than being interpolated
  // raw: it can contain build output, which is attacker-influenced.
  const blocks = errs.map((e, i) => `
    <details class="block">
      <summary>Attempt ${i + 1} failure</summary>
      <pre class="err-pre">${esc(e)}</pre>
    </details>`).join('');

  return `
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Failures</h2>
        <span class="card-meta">${errs.length} attempt(s)</span>
      </div>
      ${j.error && !errs.length ? `<pre class="err-pre">${esc(j.error)}</pre>` : ''}
      ${blocks}
    </div>`;
}

// ── Shared ───────────────────────────────────────────────────────────────────

export function emptyState(icon, title, hint) {
  return `
    <div class="empty">
      <div class="empty-icon">${esc(icon)}</div>
      <div>${esc(title)}</div>
      ${hint ? `<div class="empty-hint">${esc(hint)}</div>` : ''}
    </div>`;
}
