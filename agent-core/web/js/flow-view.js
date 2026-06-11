/**
 * flow-view.js — Hardware-centric pub/sub data-flow SVG diagram.
 *
 * Nodes: topic (pill, purple), hardware nodes (sensor/actuator/processor, tech aesthetic),
 *        agentcore (rect, accent, always present).
 * Edges: topic→node and node→topic. All topics feed Agent Core.
 * Click: topic node → show live stream; hardware node → show node info.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

// Layout constants
const TOPIC_W  = 160;
const TOPIC_H  = 28;
const NODE_W   = 148;
const NODE_H   = 56;
const AC_W     = 160;
const AC_H     = 60;
const COL_GAP  = 90;   // horizontal gap between columns
const ROW_GAP  = 40;   // vertical gap between nodes/topics in same column

let _mcps      = [];   // from /api/mcp
let _onTopicClick = null;  // callback(topic, format)
let _onNodeClick  = null;  // callback(mcp)
let _svg = null;
let _tooltip = null;

// ── Public API ───────────────────────────────────────────────────────────────

export function initFlowView(svgEl, opts = {}) {
  _svg = svgEl;
  _onTopicClick = opts.onTopicClick || null;
  _onNodeClick  = opts.onNodeClick  || null;

  // Create tooltip div
  _tooltip = document.createElement('div');
  _tooltip.id = 'flow-tooltip';
  _tooltip.classList.add('hidden');
  document.body.appendChild(_tooltip);

  return { refresh };
}

export async function refresh() {
  try {
    const res  = await fetch('/api/mcp');
    const json = await res.json();
    _mcps = json.data || [];
  } catch {
    _mcps = [];
  }
  _render();
  return _mcps;
}

// ── Classification ────────────────────────────────────────────────────────────

function _classifyDriver(mcp) {
  const hasOut = (mcp.topic_out || []).filter(t => t.topic).length > 0;
  const hasIn  = (mcp.topic_in  || []).filter(t => t.topic).length > 0;
  if (hasOut && hasIn)  return 'processor';
  if (hasOut)           return 'sensor';
  if (hasIn)            return 'actuator';
  return 'none';
}

const _HW_LABELS = {
  sensor:    'SENSOR',
  actuator:  'ACTUATOR',
  processor: 'PROC',
  none:      'MCP',
};

// ── Graph Build ───────────────────────────────────────────────────────────────

function _buildGraph(mcps) {
  const topicMap = {};  // topic_path → { path, format, sources: Set, sinks: Set }
  const nodes    = [];  // { id, name, hwType, mcp, topicIn[], topicOut[] }

  for (const mcp of mcps) {
    const node = {
      id:       mcp.id,
      name:     mcp.server_name || mcp.name,
      hwType:   _classifyDriver(mcp),
      mcp,
      topicIn:  (mcp.topic_in  || []).map(t => t.topic).filter(Boolean),
      topicOut: (mcp.topic_out || []).map(t => t.topic).filter(Boolean),
      online:   mcp.online,
    };
    nodes.push(node);

    for (const t of mcp.topic_out || []) {
      if (!t.topic) continue;
      if (!topicMap[t.topic]) topicMap[t.topic] = { path: t.topic, format: t.format || '', sources: new Set(), sinks: new Set() };
      topicMap[t.topic].sources.add(mcp.id);
    }
    for (const t of mcp.topic_in || []) {
      if (!t.topic) continue;
      if (!topicMap[t.topic]) topicMap[t.topic] = { path: t.topic, format: '', sources: new Set(), sinks: new Set() };
      topicMap[t.topic].sinks.add(mcp.id);
    }
  }

  const topics = Object.values(topicMap);
  return { nodes, topics };
}

// ── Layout ────────────────────────────────────────────────────────────────────

/**
 * Column assignment:
 *  col 0: sensor nodes (topic_out only)
 *  col 2: processor nodes (both topic_in and topic_out) + uncategorized
 *  col 4: actuator nodes (topic_in only)
 *  Topics: placed between the col of their source and the col of their sinks
 *  Agent Core: far right
 */
function _layout(nodes, topics) {
  // Assign column to nodes
  for (const n of nodes) {
    if (n.hwType === 'sensor')         n.col = 0;
    else if (n.hwType === 'processor') n.col = 2;
    else if (n.hwType === 'actuator')  n.col = 4;
    else                               n.col = 2;
  }

  // Topics get col = between source col and sink col
  for (const t of topics) {
    const sourceCols = [...t.sources].map(id => (nodes.find(n => n.id === id) || {col: 0}).col);
    const sinkCols   = [...t.sinks  ].map(id => (nodes.find(n => n.id === id) || {col: 4}).col);
    const maxSrc = sourceCols.length ? Math.max(...sourceCols) : 0;
    const minSnk = sinkCols.length   ? Math.min(...sinkCols)   : 6;
    t.col = maxSrc + 1;
    if (t.col >= minSnk) t.col = maxSrc;
  }

  // Compute x for each col
  const COL_WIDTHS = {
    0: NODE_W, 1: TOPIC_W, 2: NODE_W, 3: TOPIC_W,
    4: NODE_W, 5: TOPIC_W, 6: AC_W,
  };
  let colX = {};
  let x = 40;
  for (let c = 0; c <= 7; c++) {
    colX[c] = x;
    x += (COL_WIDTHS[c] || TOPIC_W) + COL_GAP;
  }

  // Agent Core: far right, centered vertically
  const acX = x;
  const acNode = { id: '__agentcore__', name: 'Agent Core', hwType: 'agentcore', col: 99 };

  // Assign rows within each column
  const colRows = {};
  const allItems = [...nodes, ...topics];
  for (const item of allItems) {
    const c = item.col;
    if (!colRows[c]) colRows[c] = 0;
    item.row = colRows[c]++;
  }

  // Compute y for each item
  const topY = 40;
  const allCols = [...new Set(allItems.map(i => i.col))];
  const maxRows = {};
  for (const c of allCols) maxRows[c] = colRows[c] || 0;
  const maxTotalRows = Math.max(1, ...Object.values(maxRows));
  const totalH = maxTotalRows * (NODE_H + ROW_GAP);

  for (const item of allItems) {
    const c   = item.col;
    const r   = item.row;
    const col = maxRows[c] || 1;
    const isTopic = !!item.path;
    const h = isTopic ? TOPIC_H : NODE_H;
    const w = isTopic ? TOPIC_W : NODE_W;
    const colH = col * (NODE_H + ROW_GAP);
    const startY = topY + (totalH - colH) / 2;
    const y = startY + r * (NODE_H + ROW_GAP);
    item.x = colX[c] || 40;
    item.y = y;
    item.w = w;
    item.h = h;
  }

  // Agent Core center Y
  acNode.x = acX;
  acNode.y = topY + totalH / 2 - AC_H / 2;
  acNode.w = AC_W;
  acNode.h = AC_H;

  return { nodes, topics, acNode, svgW: acX + AC_W + 40, svgH: totalH + topY * 2 };
}

// ── Render ────────────────────────────────────────────────────────────────────

function _render() {
  _svg.innerHTML = '';

  if (!_mcps.length) {
    const empty = document.getElementById('flow-empty');
    if (empty) empty.classList.remove('hidden');
    return;
  }
  const empty = document.getElementById('flow-empty');
  if (empty) empty.classList.add('hidden');

  const { nodes, topics } = _buildGraph(_mcps);
  const layout = _layout(nodes, topics);
  const { acNode, svgW, svgH } = layout;

  _svg.setAttribute('viewBox', `0 0 ${svgW} ${svgH}`);
  _svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

  // Defs: arrowheads + patterns + gradients
  const defs = _el('defs');
  defs.innerHTML = `
    <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#666"/>
    </marker>
    <marker id="arrow-dim" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#666" opacity="0.35"/>
    </marker>

    <!-- Grid dot pattern background -->
    <pattern id="grid-pattern" width="20" height="20" patternUnits="userSpaceOnUse">
      <circle cx="1" cy="1" r="0.7" fill="rgba(0,0,0,0.05)"/>
    </pattern>

    <!-- Sensor glow gradient -->
    <linearGradient id="grad-sensor" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0891B2" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#0E7490" stop-opacity="0.04"/>
    </linearGradient>
    <!-- Actuator glow gradient -->
    <linearGradient id="grad-actuator" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#D97706" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#B45309" stop-opacity="0.04"/>
    </linearGradient>
    <!-- Processor glow gradient -->
    <linearGradient id="grad-processor" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#6D28D9" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#5B21B6" stop-opacity="0.04"/>
    </linearGradient>`;
  _svg.appendChild(defs);

  // Grid background
  const bgRect = _el('rect');
  bgRect.setAttribute('width',  svgW);
  bgRect.setAttribute('height', svgH);
  bgRect.setAttribute('fill', 'url(#grid-pattern)');
  _svg.appendChild(bgRect);

  // Draw edges first (below nodes)
  const edgeGroup = _el('g');
  _svg.appendChild(edgeGroup);

  const nodeById = {};
  for (const n of nodes) nodeById[n.id] = n;

  for (const topic of topics) {
    for (const srcId of topic.sources) {
      const src = nodeById[srcId];
      if (!src) continue;
      edgeGroup.appendChild(_drawEdge(src, topic, false));
    }
    for (const sinkId of topic.sinks) {
      const sink = nodeById[sinkId];
      if (!sink) continue;
      edgeGroup.appendChild(_drawEdge(topic, sink, false));
    }
    // All topics feed Agent Core as context
    edgeGroup.appendChild(_drawEdge(topic, acNode, true));
  }

  // Draw topic nodes
  for (const topic of topics) {
    _svg.appendChild(_drawTopicNode(topic));
  }

  // Draw hardware nodes
  for (const node of nodes) {
    _svg.appendChild(_drawHardwareNode(node));
  }

  // Draw Agent Core node
  _svg.appendChild(_drawAgentCoreNode(acNode));
}

function _drawEdge(from, to, isInspector) {
  const g = _el('g');
  g.classList.add('flow-edge');
  if (isInspector) g.classList.add('inspector');

  const x1 = from.x + from.w;
  const y1 = from.y + from.h / 2;
  const x2 = to.x;
  const y2 = to.y + to.h / 2;

  const dx = (x2 - x1) * 0.5;
  const pathD = `M${x1},${y1} C${x1+dx},${y1} ${x2-dx},${y2} ${x2},${y2}`;

  const path = _el('path');
  path.setAttribute('d', pathD);
  path.setAttribute('marker-end', isInspector ? 'url(#arrow-dim)' : 'url(#arrow)');
  g.appendChild(path);
  return g;
}

function _drawTopicNode(topic) {
  const g = _el('g');
  g.classList.add('flow-topic');
  g.setAttribute('transform', `translate(${topic.x},${topic.y})`);
  g.style.cursor = 'pointer';

  const rx = TOPIC_H / 2;
  const rect = _el('rect');
  rect.setAttribute('width',  TOPIC_W);
  rect.setAttribute('height', TOPIC_H);
  rect.setAttribute('rx', rx);
  g.appendChild(rect);

  const label = _shortTopicLabel(topic.path);
  const text = _el('text');
  text.setAttribute('x', TOPIC_W / 2);
  text.setAttribute('y', TOPIC_H / 2 + 4);
  text.setAttribute('text-anchor', 'middle');
  text.textContent = label;
  g.appendChild(text);

  _addTooltip(g, topic.path, topic.format ? `format: ${topic.format}` : '');
  g.addEventListener('click', () => {
    _onTopicClick?.(topic.path, topic.format);
  });

  return g;
}

function _drawHardwareNode(node) {
  const W = NODE_W, H = NODE_H, notch = 10;
  const g = _el('g');
  g.classList.add('flow-hw-node', `flow-hw-${node.hwType}`);
  if (node.online === true) g.classList.add('hw-online');
  g.setAttribute('transform', `translate(${node.x},${node.y})`);
  g.style.cursor = 'pointer';

  // Gradient background fill rect (under the polygon stroke)
  const gradId = `grad-${node.hwType}`;
  const bgFill = _el('rect');
  bgFill.setAttribute('width', W);
  bgFill.setAttribute('height', H);
  bgFill.setAttribute('rx', 4);
  bgFill.setAttribute('fill', `url(#${gradId})`);
  g.appendChild(bgFill);

  // Cut-corner polygon (top-right corner notched)
  const poly = _el('polygon');
  poly.setAttribute('points', `0,0 ${W-notch},0 ${W},${notch} ${W},${H} 0,${H}`);
  poly.setAttribute('rx', 4);
  g.appendChild(poly);

  // Notch accent line (diagonal cut line)
  const notchLine = _el('line');
  notchLine.setAttribute('x1', W - notch);
  notchLine.setAttribute('y1', 0);
  notchLine.setAttribute('x2', W);
  notchLine.setAttribute('y2', notch);
  notchLine.style.stroke = 'currentColor';
  notchLine.style.strokeWidth = '0.8';
  notchLine.style.opacity = '0.3';
  g.appendChild(notchLine);

  // Type badge (top-left)
  const BW = node.hwType === 'actuator' ? 58 : 48, BH = 14;
  const badgeRect = _el('rect');
  badgeRect.setAttribute('x', 6);
  badgeRect.setAttribute('y', 5);
  badgeRect.setAttribute('width', BW);
  badgeRect.setAttribute('height', BH);
  badgeRect.setAttribute('rx', 2);
  badgeRect.classList.add('hw-badge-rect');
  g.appendChild(badgeRect);

  const badgeText = _el('text');
  badgeText.setAttribute('x', 6 + BW / 2);
  badgeText.setAttribute('y', 5 + BH - 3);
  badgeText.setAttribute('text-anchor', 'middle');
  badgeText.classList.add('hw-badge-text');
  badgeText.textContent = _HW_LABELS[node.hwType] || 'MCP';
  g.appendChild(badgeText);

  // Status dot (bottom-right, inside notch area)
  const dotR = 4;
  const dot = _el('circle');
  dot.setAttribute('cx', W - 8);
  dot.setAttribute('cy', H - 8);
  dot.setAttribute('r',  dotR);
  const statusCls = node.online === true ? 'status-online' : node.online === false ? 'status-offline' : 'status-pending';
  dot.classList.add(statusCls);
  g.appendChild(dot);

  // Node name
  const nameText = _el('text');
  nameText.setAttribute('x', 8);
  nameText.setAttribute('y', H / 2 + 6);
  nameText.textContent = node.name.length > 15 ? node.name.slice(0, 14) + '…' : node.name;
  g.appendChild(nameText);

  // Topic count sub-label
  const topicCount = (node.topicIn?.length || 0) + (node.topicOut?.length || 0);
  if (topicCount > 0) {
    const sub = _el('text');
    sub.setAttribute('x', 8);
    sub.setAttribute('y', H - 8);
    sub.style.fontSize = '9px';
    sub.style.fill = 'var(--text-dim)';
    sub.textContent = `${topicCount} topic${topicCount > 1 ? 's' : ''}`;
    g.appendChild(sub);
  }

  const toolList = (node.mcp.tools || []).map(t => typeof t === 'string' ? t : t.name).join(', ') || '—';
  _addTooltip(g, node.name, `${node.hwType}\n${node.mcp.url || ''}\n工具: ${toolList}`);
  g.addEventListener('click', () => _onNodeClick?.(node.mcp));

  return g;
}

function _drawAgentCoreNode(acNode) {
  const g = _el('g');
  g.classList.add('flow-agentcore');
  g.setAttribute('transform', `translate(${acNode.x},${acNode.y})`);

  const rect = _el('rect');
  rect.setAttribute('width',  AC_W);
  rect.setAttribute('height', AC_H);
  rect.setAttribute('rx', 8);
  g.appendChild(rect);

  const label = _el('text');
  label.setAttribute('x', AC_W / 2);
  label.setAttribute('y', AC_H / 2 + 5);
  label.setAttribute('text-anchor', 'middle');
  label.textContent = 'Agent Core';
  g.appendChild(label);

  return g;
}

// ── Tooltip ───────────────────────────────────────────────────────────────────

function _addTooltip(el, title, detail) {
  el.addEventListener('mouseenter', (e) => {
    _tooltip.classList.remove('hidden');
    _tooltip.innerHTML = `<div class="tt-title">${title}</div>${detail ? `<div class="tt-row">${detail.replace(/\n/g, '<br>')}</div>` : ''}`;
    _positionTooltip(e);
  });
  el.addEventListener('mousemove', _positionTooltip);
  el.addEventListener('mouseleave', () => _tooltip.classList.add('hidden'));
}

function _positionTooltip(e) {
  const pad = 12;
  let x = e.clientX + pad;
  let y = e.clientY + pad;
  const w = _tooltip.offsetWidth  || 200;
  const h = _tooltip.offsetHeight || 60;
  if (x + w > window.innerWidth)  x = e.clientX - w - pad;
  if (y + h > window.innerHeight) y = e.clientY - h - pad;
  _tooltip.style.left = x + 'px';
  _tooltip.style.top  = y + 'px';
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _el(tag) {
  return document.createElementNS(SVG_NS, tag);
}

function _shortTopicLabel(path) {
  const parts = path.split('/').filter(Boolean);
  if (parts.length <= 2) return path;
  return '/' + parts.slice(-2).join('/');
}

/** Flash an edge to indicate live activity on a topic */
export function flashTopic(topicPath) {
  const edges = _svg?.querySelectorAll('.flow-edge');
  if (!edges) return;
  for (const e of edges) {
    if (!e.classList.contains('inspector')) {
      e.classList.add('active');
      setTimeout(() => e.classList.remove('active'), 600);
      break;
    }
  }
}
