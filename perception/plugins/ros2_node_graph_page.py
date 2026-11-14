"""Self-contained browser page for the ROS 2 node-graph card.

No CDN, no framework: the robot may be on an isolated network and the page must
work as soon as the SSH tunnel is up.  The page only reads ``api/graph``; every
node/topic name is rendered through ``textContent``/canvas text, never HTML.
"""

PAGE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ROS 2 Node Graph</title>
<style>
:root{--bg:#0f141b;--panel:#161d27;--line:#2a3646;--text:#dbe4ef;--muted:#8c9bad;--accent:#f0a24a;}
*{box-sizing:border-box}
html,body{height:100%;margin:0;background:var(--bg);color:var(--text);font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
body{display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:center;gap:16px;padding:10px 14px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
#stats{color:var(--muted);white-space:nowrap}
#status{color:var(--accent);white-space:nowrap}
header label{color:var(--muted);display:flex;align-items:center;gap:6px;cursor:pointer}
button{background:#223041;color:var(--text);border:1px solid var(--line);border-radius:5px;padding:4px 10px;cursor:pointer}
button:hover{border-color:var(--accent)}
main{flex:1;display:flex;min-height:0}
#stage{flex:1;position:relative;min-width:0}
canvas{display:block;width:100%;height:100%;cursor:grab}
canvas.dragging{cursor:grabbing}
aside{width:340px;background:var(--panel);border-left:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
#search{margin:10px;padding:7px 9px;background:#0d131a;border:1px solid var(--line);border-radius:5px;color:var(--text);outline:none}
#search:focus{border-color:var(--accent)}
#details{padding:0 12px 10px;color:var(--muted);overflow:auto;max-height:42%;border-bottom:1px solid var(--line)}
#details h2{font-size:12px;color:var(--text);margin:4px 0 6px;word-break:break-all}
#details .row{margin:3px 0;word-break:break-all}
#details .k{color:#7e8ea3}
#details code{color:#cfe0f2}
#summary{flex:1;margin:0;padding:10px 12px;overflow:auto;white-space:pre-wrap;word-break:break-word;color:#aebccd;font-size:11.5px}
#tooltip{position:fixed;pointer-events:none;background:#0b1118;border:1px solid var(--line);border-radius:5px;padding:5px 7px;color:var(--text);max-width:520px;display:none;z-index:10;font-size:11.5px;word-break:break-all}
.legend{color:var(--muted);display:flex;gap:14px;align-items:center}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;vertical-align:1px}
.sq{display:inline-block;width:8px;height:8px;background:var(--accent);margin-right:4px;vertical-align:1px}
@media (max-width:900px){aside{width:270px}}
</style>
</head>
<body>
<header>
  <h1>ROS 2 Node Graph</h1>
  <div id="stats">—</div>
  <div id="status"></div>
  <div class="legend"><span><i class="dot" style="background:#5aa9e6"></i>node</span><span><i class="sq"></i>topic</span></div>
  <span style="flex:1"></span>
  <label><input id="auto" type="checkbox" checked> 自动刷新</label>
  <button id="refresh">刷新</button>
</header>
<main>
  <div id="stage"><canvas id="graph"></canvas></div>
  <aside>
    <input id="search" placeholder="搜索节点 / 话题（Esc 清除）">
    <div id="details"><div class="row">点击节点或话题查看连接关系。</div></div>
    <pre id="summary"></pre>
  </aside>
</main>
<div id="tooltip"></div>
<script>
"use strict";
const canvas = document.getElementById('graph');
const ctx = canvas.getContext('2d');
const statsEl = document.getElementById('stats');
const statusEl = document.getElementById('status');
const detailsEl = document.getElementById('details');
const summaryEl = document.getElementById('summary');
const searchEl = document.getElementById('search');
const autoEl = document.getElementById('auto');
const tooltip = document.getElementById('tooltip');

let DATA = null, ITEMS = [], EDGES = [], MAP = new Map();
let selected = null, hovered = null, dragging = null, panning = null;
let zoom = 1, panX = 0, panY = 0, step = 0, query = '';

const dpr = () => window.devicePixelRatio || 1;
function resize() {
  const box = canvas.parentElement.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(box.width * dpr()));
  canvas.height = Math.max(1, Math.floor(box.height * dpr()));
  if (!panX && !panY) { panX = box.width / 2; panY = box.height / 2; }
  draw();
}
window.addEventListener('resize', resize);

function hash(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function nodeColor(ns) {
  const hue = hash(ns || '/') % 360;
  return 'hsl(' + hue + ' 62% 62%)';
}
function shortLabel(text) {
  const t = text.startsWith('/') ? text.slice(1) : text;
  return t.length > 34 ? t.slice(0, 32) + '…' : t;
}

function build(data) {
  const previous = new Map(ITEMS.map(item => [item.id, item]));
  DATA = data; ITEMS = []; EDGES = []; MAP = new Map();
  const add = (item) => {
    if (MAP.has(item.id)) return MAP.get(item.id);
    const old = previous.get(item.id);
    item.x = old ? old.x : 0; item.y = old ? old.y : 0;
    item.vx = 0; item.vy = 0; item.deg = 0;
    ITEMS.push(item); MAP.set(item.id, item);
    return item;
  };
  const link = (aId, bId) => {
    const a = MAP.get(aId), b = MAP.get(bId);
    if (!a || !b) return;
    EDGES.push({a, b}); a.deg++; b.deg++;
  };
  for (const node of (data.nodes || [])) {
    add({id: 'n:' + node.name, label: node.name, kind: 'node', ns: node.namespace || '/', internal: !!node.internal});
  }
  for (const topic of (data.topics || [])) {
    add({id: 't:' + topic.name, label: topic.name, kind: 'topic', types: topic.types || []});
  }
  for (const topic of (data.topics || [])) {
    const tid = 't:' + topic.name;
    for (const pub of (topic.publishers || [])) {
      add({id: 'n:' + pub, label: pub, kind: 'node', ns: pub.split('/').slice(0, -1).join('/') || '/'});
      link('n:' + pub, tid);
    }
    for (const sub of (topic.subscribers || [])) {
      add({id: 'n:' + sub, label: sub, kind: 'node', ns: sub.split('/').slice(0, -1).join('/') || '/'});
      link(tid, 'n:' + sub);
    }
  }
  const fresh = ITEMS.filter(item => !previous.has(item.id));
  fresh.forEach((item, index) => {
    const angle = (index / Math.max(1, fresh.length)) * Math.PI * 2;
    const radius = item.kind === 'topic' ? 90 : 240;
    item.x = Math.cos(angle) * radius + (Math.random() - 0.5) * 24;
    item.y = Math.sin(angle) * radius + (Math.random() - 0.5) * 24;
  });
  step = 0;
  if (selected) selected = MAP.get(selected.id) || null;
  hovered = null;
  updateStats(data); updateDetails(); summaryEl.textContent = data.summary || '';
  draw();
}

function updateStats(data) {
  const stats = data.stats || {};
  statsEl.textContent = (stats.nodes || 0) + ' nodes · ' + (stats.topics || 0) +
    ' topics · ' + (stats.links || 0) + ' links' +
    (data.truncated ? ' · truncated' : '');
  statusEl.textContent = data.state && data.state !== 'running' ? data.state : '';
}

function simulate() {
  for (const item of ITEMS) { item.fx = 0; item.fy = 0; }
  for (let i = 0; i < ITEMS.length; i++) {
    const a = ITEMS[i];
    for (let j = i + 1; j < ITEMS.length; j++) {
      const b = ITEMS[j];
      let dx = b.x - a.x, dy = b.y - a.y;
      let d2 = dx * dx + dy * dy;
      if (d2 < 1) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 1; }
      const d = Math.sqrt(d2);
      const force = 5200 / d2;
      const fx = (dx / d) * force, fy = (dy / d) * force;
      a.fx -= fx; a.fy -= fy; b.fx += fx; b.fy += fy;
    }
    a.fx += -a.x * 0.0016; a.fy += -a.y * 0.0016;
  }
  for (const edge of EDGES) {
    const dx = edge.b.x - edge.a.x, dy = edge.b.y - edge.a.y;
    const d = Math.hypot(dx, dy) || 1;
    const force = (d - 95) * 0.004;
    const fx = (dx / d) * force, fy = (dy / d) * force;
    edge.a.fx += fx; edge.a.fy += fy;
    edge.b.fx -= fx; edge.b.fy -= fy;
  }
  for (const item of ITEMS) {
    if (item === dragging) continue;
    item.vx = (item.vx + item.fx) * 0.86;
    item.vy = (item.vy + item.fy) * 0.86;
    const speed = Math.hypot(item.vx, item.vy);
    if (speed > 8) { item.vx *= 8 / speed; item.vy *= 8 / speed; }
    item.x += item.vx; item.y += item.vy;
  }
}

function matches(item) {
  return query && item.label.toLowerCase().includes(query);
}

function drawArrow(from, to) {
  const dx = to.x - from.x, dy = to.y - from.y;
  const d = Math.hypot(dx, dy) || 1;
  const ux = dx / d, uy = dy / d;
  const tipX = to.x - ux * 7, tipY = to.y - uy * 7;
  const size = 4.5;
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - ux * size - uy * size * 0.6, tipY - uy * size + ux * size * 0.6);
  ctx.lineTo(tipX - ux * size + uy * size * 0.6, tipY - uy * size - ux * size * 0.6);
  ctx.closePath();
  ctx.fillStyle = 'rgba(150,175,205,0.45)';
  ctx.fill();
}

function draw() {
  const width = canvas.width / dpr(), height = canvas.height / dpr();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  ctx.translate(panX, panY);
  ctx.scale(zoom, zoom);

  for (const edge of EDGES) {
    const dim = query && !matches(edge.a) && !matches(edge.b);
    ctx.beginPath();
    ctx.moveTo(edge.a.x, edge.a.y);
    ctx.lineTo(edge.b.x, edge.b.y);
    ctx.strokeStyle = dim ? 'rgba(140,170,210,0.07)' : 'rgba(140,170,210,0.20)';
    ctx.lineWidth = 1 / zoom;
    ctx.stroke();
    if (!dim) drawArrow(edge.a, edge.b);
  }

  for (const item of ITEMS) {
    const isNode = item.kind === 'node';
    const size = isNode ? 5 + Math.min(6, item.deg * 0.8) : 3.5 + Math.min(4, item.deg * 0.5);
    const dim = query && !matches(item);
    const active = item === selected || item === hovered;
    ctx.globalAlpha = dim ? 0.22 : 1;
    if (isNode) {
      ctx.beginPath();
      ctx.arc(item.x, item.y, size, 0, Math.PI * 2);
      ctx.fillStyle = item.internal ? '#68788b' : nodeColor(item.ns);
      ctx.fill();
    } else {
      ctx.fillStyle = '#f0a24a';
      ctx.fillRect(item.x - size, item.y - size, size * 2, size * 2);
    }
    if (active) {
      ctx.lineWidth = 2 / zoom;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    }
    if (zoom >= 0.85 || active || (query && matches(item))) {
      ctx.font = (11 / zoom) + 'px ui-monospace, Meiryo, monospace';
      ctx.fillStyle = dim ? 'rgba(219,228,239,0.25)' : '#c8d4e2';
      ctx.textBaseline = 'middle';
      ctx.fillText(shortLabel(item.label), item.x + size + 3 / zoom, item.y);
    }
    ctx.globalAlpha = 1;
  }
  ctx.restore();
}

function worldPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left - panX) / zoom,
    y: (event.clientY - rect.top - panY) / zoom,
  };
}
function pick(point) {
  let best = null, bestDistance = 16 / zoom;
  for (const item of ITEMS) {
    const d = Math.hypot(item.x - point.x, item.y - point.y);
    if (d < bestDistance) { best = item; bestDistance = d; }
  }
  return best;
}

canvas.addEventListener('mousemove', (event) => {
  if (dragging) {
    const p = worldPoint(event);
    dragging.x = p.x; dragging.y = p.y; dragging.vx = 0; dragging.vy = 0;
    step = Math.min(step, 300);
    draw();
    return;
  }
  if (panning) {
    panX = panning.px + (event.clientX - panning.x);
    panY = panning.py + (event.clientY - panning.y);
    draw();
    return;
  }
  const hit = pick(worldPoint(event));
  if (hit !== hovered) { hovered = hit; draw(); }
  if (hit) {
    tooltip.style.display = 'block';
    tooltip.textContent = hit.label + (hit.kind === 'topic' && hit.types.length ? '  [' + hit.types.join(', ') + ']' : '');
    tooltip.style.left = Math.min(window.innerWidth - 320, event.clientX + 12) + 'px';
    tooltip.style.top = (event.clientY + 14) + 'px';
  } else {
    tooltip.style.display = 'none';
  }
});
canvas.addEventListener('mousedown', (event) => {
  const hit = pick(worldPoint(event));
  if (hit) { dragging = hit; canvas.classList.add('dragging'); }
  else { panning = {x: event.clientX, y: event.clientY, px: panX, py: panY}; }
  event.preventDefault();
});
window.addEventListener('mouseup', () => {
  if (dragging) { dragging = null; step = Math.min(step, 350); }
  panning = null;
  canvas.classList.remove('dragging');
});
canvas.addEventListener('click', (event) => {
  const hit = pick(worldPoint(event));
  if (hit) { selected = hit; updateDetails(); draw(); }
  else if (selected) { selected = null; updateDetails(); draw(); }
});
canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = event.clientX - rect.left, my = event.clientY - rect.top;
  const wx = (mx - panX) / zoom, wy = (my - panY) / zoom;
  const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
  zoom = Math.max(0.25, Math.min(4, zoom * factor));
  panX = mx - wx * zoom;
  panY = my - wy * zoom;
  draw();
}, {passive: false});
canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; hovered = null; draw(); });

function addRow(parent, key, value) {
  const row = document.createElement('div');
  row.className = 'row';
  const k = document.createElement('span'); k.className = 'k'; k.textContent = key + ': ';
  const v = document.createElement('code'); v.textContent = value;
  row.append(k, v); parent.appendChild(row);
}
function updateDetails() {
  detailsEl.textContent = '';
  if (!selected) {
    const hint = document.createElement('div');
    hint.className = 'row';
    hint.textContent = '点击节点或话题查看连接关系。';
    detailsEl.appendChild(hint);
    return;
  }
  const title = document.createElement('h2');
  title.textContent = selected.label;
  detailsEl.appendChild(title);
  addRow(detailsEl, '类型', selected.kind === 'node' ? 'ROS node' : 'topic' + (selected.types.length ? ' / ' + selected.types.join(', ') : ''));
  addRow(detailsEl, '连接数', String(selected.deg));
  if (selected.kind === 'node') {
    const published = (DATA.topics || []).filter(t => (t.publishers || []).includes(selected.label)).map(t => t.name);
    const subscribed = (DATA.topics || []).filter(t => (t.subscribers || []).includes(selected.label)).map(t => t.name);
    addRow(detailsEl, '发布', published.join(', ') || '—');
    addRow(detailsEl, '订阅', subscribed.join(', ') || '—');
  } else {
    const topic = (DATA.topics || []).find(t => t.name === selected.label);
    if (topic) {
      addRow(detailsEl, '发布者', (topic.publishers || []).join(', ') || '—');
      addRow(detailsEl, '订阅者', (topic.subscribers || []).join(', ') || '—');
    }
  }
}

searchEl.addEventListener('input', () => { query = searchEl.value.trim().toLowerCase(); draw(); });
searchEl.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') { searchEl.value = ''; query = ''; draw(); }
});

async function load() {
  statusEl.textContent = '加载中…';
  try {
    const response = await fetch('api/graph?ts=' + Date.now(), {cache: 'no-store'});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const data = await response.json();
    if (!data.stats) {
      statusEl.textContent = data.error || '卡片未启动';
      return;
    }
    build(data);
    statusEl.textContent = data.truncated ? '已截断' : '';
  } catch (error) {
    statusEl.textContent = '无法连接卡片：' + error.message;
  }
}

document.getElementById('refresh').addEventListener('click', load);
function ticker() {
  if (step < 420) {
    simulate();
    if (step < 160) simulate();
    step++;
    draw();
  }
  requestAnimationFrame(ticker);
}
setInterval(() => { if (autoEl.checked) load(); }, 2500);
resize();
load();
requestAnimationFrame(ticker);
</script>
</body>
</html>
"""
