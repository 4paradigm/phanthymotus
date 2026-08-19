/** api.js — fetch helpers and formatting utilities.
 *
 * agent-core has no shared utils module; every module redeclares its own `_esc`
 * and formatters, and several then interpolate raw values into innerHTML anyway.
 * This dashboard centralises them instead, because nearly everything it renders
 * is influenced by whoever opened the PR — branch names, build output, error
 * text, and LLM review that quotes the diff.
 */

// ── HTTP ─────────────────────────────────────────────────────────────────────

async function _json(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json();
}

export function getStatus() {
  return _json('/api/status');
}

export function getJobs({ limit = 50, offset = 0, status = '', repo = '' } = {}) {
  const q = new URLSearchParams({ limit, offset });
  if (status) q.set('status', status);
  if (repo) q.set('repo', repo);
  return _json(`/api/jobs?${q}`);
}

export function getJob(id) {
  return _json(`/api/jobs/${encodeURIComponent(id)}`);
}

export function getLog(jobId, idx, offset = 0) {
  const q = new URLSearchParams({ offset });
  return _json(`/api/jobs/${encodeURIComponent(jobId)}/log/${idx}?${q}`);
}

export function getReviewTrace(jobId, offset = 0) {
  const q = new URLSearchParams({ offset });
  return _json(`/api/jobs/${encodeURIComponent(jobId)}/review-trace?${q}`);
}

// ── Escaping ─────────────────────────────────────────────────────────────────

/** Escape for interpolation into innerHTML. */
export function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Render a markdown subset as HTML.
 *
 * Escaping happens FIRST, then the pattern substitutions run over already-safe
 * text. That order is what makes this safe — substituting first would let
 * attacker-controlled markup through, and the review text quotes the PR's diff.
 * A full markdown parser would need a sanitizer to match this guarantee.
 */
export function renderMarkdown(src) {
  const lines = esc(src).split('\n');
  const out = [];
  let inList = false;
  let table = [];

  const closeList = () => {
    if (inList) { out.push('</ul>'); inList = false; }
  };
  // Pipe tables are common in PR descriptions here, and a real table parser is
  // more machinery (and more escaping risk) than this needs. Consecutive pipe
  // lines are emitted as a preformatted block instead: the columns line up, and
  // nothing is reinterpreted as markup.
  const closeTable = () => {
    if (table.length) {
      out.push(`<pre class="md-table">${table.join('\n')}</pre>`);
      table = [];
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (/^\s*\|.*\|\s*$/.test(line)) {
      closeList();
      table.push(line.trim());
      continue;
    }
    closeTable();

    const heading = line.match(/^#{1,6}\s+(.*)$/);
    if (heading) {
      closeList();
      out.push(`<h3>${_inline(heading[1])}</h3>`);
      continue;
    }

    const item = line.match(/^\s*[-*]\s+(.*)$/);
    if (item) {
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push(`<li>${_inline(item[1])}</li>`);
      continue;
    }

    if (!line) { closeList(); continue; }

    closeList();
    out.push(`<p>${_inline(line)}</p>`);
  }
  closeList();
  closeTable();
  return out.join('');
}

/** Inline emphasis over already-escaped text. */
function _inline(s) {
  return s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

// ── Formatting ───────────────────────────────────────────────────────────────

/** Seconds as a compact duration. */
export function fmtDuration(sec) {
  if (sec == null) return '—';
  const s = Math.max(0, Math.floor(sec));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

/** Unix seconds as a local timestamp. */
export function fmtTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
         `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/** Unix seconds as "3m ago". */
export function fmtRelative(ts) {
  if (!ts) return '—';
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/** ISO8601 string as "3m ago" — the poller reports timestamps this way. */
export function fmtRelativeIso(iso) {
  if (!iso) return 'never';
  const t = Date.parse(iso);
  return Number.isNaN(t) ? 'never' : fmtRelative(t / 1000);
}

export function shortSha(sha) {
  return String(sha ?? '').slice(0, 7) || '—';
}

/** Human label for a build target row. */
export function targetLabel(br) {
  const name = br.driver_path || br.target || 'build';
  // A job can hold two perception builds — without the variant they render as
  // duplicates of each other.
  return br.variant ? `${name} (jetson-jp${br.variant})` : name;
}

/** "owner/repo" -> "repo", which is what disambiguates in this UI. */
export function shortRepo(repo) {
  return String(repo ?? '').split('/').pop() || '—';
}

export function prUrl(repo, prNumber) {
  return `https://github.com/${repo}/pull/${prNumber}`;
}
