const CARD_WIDTH = 184;
const CARD_HEIGHT = 62;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function button(label, handler) {
  const control = element('button', '', label);
  control.type = 'button';
  control.addEventListener('click', handler);
  return control;
}

function setOptions(select, options) {
  select.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
}

function cardName(card) {
  return card.toolName || card.driverName || '卡片';
}

function toolKey(mcpId, toolName) {
  return JSON.stringify([mcpId, toolName]);
}

function getCatalog(mcps) {
  return new Map(mcps.flatMap(mcp => (mcp.tools || []).flatMap(tool => {
    const item = typeof tool === 'string' ? { name: tool } : tool;
    if (!item?.name) return [];
    const named = mcp.server_name || mcp.name || mcp.id;
    return [[toolKey(mcp.id, item.name), {
      mcpId: mcp.id, toolName: item.name, driverName: named,
      type: item.type, multiInstance: !!item.multiInstance,
      topicIn: item.topic_in || [], topicOut: item.topic_out || [],
    }]];
  })));
}

export function draftLinkOptions(from, to, catalog) {
  if (!from || !to || from.id === to.id) return [];
  const fromTool = catalog.get(toolKey(from.mcpId, from.toolName));
  const toTool = catalog.get(toolKey(to.mcpId, to.toolName));
  const outputs = from.topicOutFrom !== undefined ? (from.topicOut || [])
    : from.topicOut?.length ? from.topicOut : (fromTool?.topicOut || []);
  const inputs = to.topicIn?.length ? to.topicIn : (toTool?.topicIn || []);
  const endpoints = { fromCardId: from.id, toCardId: to.id };
  const options = outputs.flatMap((out, oi) => inputs.flatMap((input, ii) => {
    if (!out?.format || out.format !== input?.format) return [];
    return [{
      list: 'connections', label: `数据 · ${out.format} · ${oi + 1} → ${ii + 1}`,
      link: { ...endpoints, fromPortIdx: String(oi), toPortIdx: String(ii), format: out.format, fromTopic: out.topic || '' },
    }];
  }));
  if (fromTool?.type === 'controller') {
    options.push({
      list: 'execConnections', label: '执行器',
      link: { ...endpoints, toToolName: to.toolName, toMcpId: to.mcpId },
    });
  }
  return options;
}

export function removeDraftCard(layout, id) {
  const remains = link => link.fromCardId !== id && link.toCardId !== id;
  layout.cards = layout.cards.filter(card => card.id !== id);
  layout.connections = layout.connections.filter(remains);
  layout.execConnections = layout.execConnections.filter(remains);
}

export function openCanvasDraftEditor({ layout, mcps, onSave, onClose, onApply }) {
  const draft = structuredClone(layout ?? { transform: { zoom: 1, tx: 0, ty: 0 } });
  draft.cards ||= [];
  draft.connections ||= [];
  draft.execConnections ||= [];
  const catalog = getCatalog(mcps);
  let saved = true;
  let nextId = 0;
  const serial = Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  const id = kind => `${kind}-${serial}-${nextId++}`;
  const origin = {
    x: Math.min(0, ...draft.cards.map(c => c.x || 0)) - 24,
    y: Math.min(0, ...draft.cards.map(c => c.y || 0)) - 24,
  };

  const dialog = element('dialog', 'canvas-drafts-editor');
  const title = element('h2', '', '编辑草稿');
  title.id = 'canvas-draft-editor-heading';
  dialog.setAttribute('aria-labelledby', title.id);
  const header = element('header', 'draft-editor-header');
  const status = element('span', 'draft-editor-status', '本地草稿 · 保存在当前浏览器');
  status.setAttribute('role', 'status');
  const apply = button('应用到共享画布', () => {
    dialog.close('apply');
    onApply(draft);
  });
  apply.disabled = layout === null;
  const back = button('返回草稿列表', () => {
    if (saved || save()) dialog.close();
  });
  header.append(title, status, apply, back);
  dialog.addEventListener('cancel', event => {
    if (!saved && !save()) event.preventDefault();
  });

  function save() {
    saved = onSave(draft);
    status.textContent = saved ? '本地草稿 · 已保存' : '草稿未保存，请检查浏览器存储';
    apply.disabled = !saved;
    return saved;
  }

  const controls = element('div', 'draft-editor-controls');
  const tools = element('select');
  tools.setAttribute('aria-label', '选择要添加的卡片');
  setOptions(tools, [...catalog].map(([key, tool]) => [key, `${tool.toolName} · ${tool.driverName}`]));
  const add = button('添加卡片', () => {
    const tool = catalog.get(tools.value);
    if (!tool) return;
    const { type, multiInstance, ...card } = tool;
    if (!multiInstance && draft.cards.some(c => c.mcpId === card.mcpId && c.toolName === card.toolName)) {
      status.textContent = '这张卡片已经在草稿中';
      return;
    }
    let row = 0;
    let x, y;
    do {
      x = (row % 3) * (CARD_WIDTH + 28);
      y = Math.floor(row / 3) * (CARD_HEIGHT + 24);
      row++;
    } while (draft.cards.some(c => Math.abs((c.x || 0) - x) < CARD_WIDTH + 6 && Math.abs((c.y || 0) - y) < CARD_HEIGHT + 6));
    draft.cards.push({ ...card, id: id('card'), x, y });
    save();
    render();
  });
  add.disabled = !catalog.size;
  controls.append(tools, add);

  const scroll = element('section', 'draft-editor-scroll');
  scroll.setAttribute('aria-label', '草稿卡片画布');
  const board = element('div', 'draft-editor-board');
  scroll.append(board);
  const wiring = element('section', 'draft-editor-wiring');
  const source = element('select');
  source.setAttribute('aria-label', '连线来源卡片');
  const target = element('select');
  target.setAttribute('aria-label', '连线目标卡片');
  const ports = element('select');
  ports.setAttribute('aria-label', '兼容的端口');
  let options = [];
  const connect = button('添加连线', () => {
    const choice = options[Number(ports.value)];
    if (!choice) return;
    const duplicate = draft[choice.list].some(link => Object.entries(choice.link)
      .every(([key, value]) => key === 'fromTopic' || String(link[key]) === String(value)));
    if (duplicate) {
      status.textContent = '这条连线已存在';
      return;
    }
    draft[choice.list].push({ id: id(choice.list === 'execConnections' ? 'exec' : 'conn'), ...choice.link });
    save();
    render();
  });
  const existing = element('ul', 'draft-editor-links');
  const updateOptions = () => {
    options = draftLinkOptions(
      draft.cards.find(c => c.id === source.value),
      draft.cards.find(c => c.id === target.value),
      catalog,
    );
    setOptions(ports, options.length ? options.map((opt, i) => [String(i), opt.label]) : [['', '无兼容端口']]);
    connect.disabled = !options.length;
  };
  source.addEventListener('change', updateOptions);
  target.addEventListener('change', updateOptions);
  const wiringControls = element('div', 'draft-editor-controls');
  wiringControls.append(source, target, ports, connect);
  wiring.append(element('h3', '', '卡片连线'), wiringControls, existing);

  const position = (card, node) => {
    node.style.left = `${(card.x || 0) - origin.x}px`;
    node.style.top = `${(card.y || 0) - origin.y}px`;
  };
  function drawLines() {
    board.querySelectorAll('.draft-editor-line').forEach(line => line.remove());
    const cards = new Map(draft.cards.map(card => [card.id, card]));
    for (const link of [...draft.connections, ...draft.execConnections]) {
      const from = cards.get(link.fromCardId);
      const to = cards.get(link.toCardId);
      if (!from || !to) continue;
      const x = (from.x || 0) - origin.x + CARD_WIDTH / 2;
      const y = (from.y || 0) - origin.y + CARD_HEIGHT / 2;
      const dx = (to.x || 0) - (from.x || 0);
      const dy = (to.y || 0) - (from.y || 0);
      const line = element('div', 'draft-editor-line');
      line.style.left = `${x}px`;
      line.style.top = `${y}px`;
      line.style.width = `${Math.hypot(dx, dy)}px`;
      line.style.transform = `rotate(${Math.atan2(dy, dx)}rad)`;
      board.prepend(line);
    }
    board.style.width = `${Math.max(0, ...draft.cards.map(c => (c.x || 0) - origin.x + CARD_WIDTH + 24))}px`;
    board.style.height = `${Math.max(0, ...draft.cards.map(c => (c.y || 0) - origin.y + CARD_HEIGHT + 24))}px`;
  }

  function draggable(grip, card, node) {
    let drag = null;
    function move(x, y) {
      card.x = Math.max(origin.x + 8, x);
      card.y = Math.max(origin.y + 8, y);
      position(card, node);
      drawLines();
    }
    grip.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      drag = { pointer: event.pointerId, x: event.clientX, y: event.clientY, startX: card.x || 0, startY: card.y || 0 };
      grip.setPointerCapture(event.pointerId);
    });
    grip.addEventListener('pointermove', event => {
      if (drag?.pointer !== event.pointerId) return;
      move(drag.startX + event.clientX - drag.x, drag.startY + event.clientY - drag.y);
    });
    grip.addEventListener('lostpointercapture', event => {
      if (drag?.pointer !== event.pointerId) return;
      if (card.x !== drag.startX || card.y !== drag.startY) save();
      drag = null;
    });
    grip.addEventListener('keydown', event => {
      const direction = { ArrowLeft: [-12, 0], ArrowRight: [12, 0], ArrowUp: [0, -12], ArrowDown: [0, 12] }[event.key];
      if (!direction) return;
      event.preventDefault();
      move((card.x || 0) + direction[0], (card.y || 0) + direction[1]);
      save();
    });
  }

  function render() {
    board.replaceChildren();
    if (!draft.cards.length) board.append(element('p', 'draft-editor-empty', '添加卡片开始编排草稿'));
    for (const card of draft.cards) {
      const node = element('article', 'draft-editor-card');
      const grip = element('button', '', cardName(card));
      grip.type = 'button';
      grip.title = '拖动或用方向键移动卡片';
      grip.className = 'draft-editor-grip';
      const remove = button('移除', () => {
        removeDraftCard(draft, card.id);
        save();
        render();
      });
      remove.setAttribute('aria-label', `从草稿移除 ${cardName(card)}`);
      node.append(grip, remove, element('span', 'draft-editor-driver', card.driverName));
      position(card, node);
      draggable(grip, card, node);
      board.append(node);
    }
    drawLines();

    const cards = draft.cards.map(card => [card.id, cardName(card)]);
    const previous = [source.value, target.value];
    setOptions(source, cards);
    setOptions(target, cards);
    if (cards.some(([key]) => key === previous[0])) source.value = previous[0];
    if (cards.some(([key]) => key === previous[1])) target.value = previous[1];
    else if (cards.length > 1) target.selectedIndex = 1;
    updateOptions();

    const names = new Map(cards);
    existing.replaceChildren();
    for (const list of ['connections', 'execConnections']) {
      draft[list].forEach((link, index) => {
        const purpose = list === 'execConnections' ? '执行器' : (link.format || '数据');
        const label = `${names.get(link.fromCardId) || '卡片'} → ${names.get(link.toCardId) || '卡片'} · ${purpose}`;
        const row = element('li', '', label);
        const remove = button('移除', () => {
          draft[list].splice(index, 1);
          save();
          render();
        });
        remove.setAttribute('aria-label', `移除连线 ${label}`);
        row.append(remove);
        existing.append(row);
      });
    }
  }

  dialog.append(header, controls, scroll, wiring);
  dialog.addEventListener('close', () => {
    dialog.remove();
    if (dialog.returnValue !== 'apply') onClose();
  }, { once: true });
  document.body.append(dialog);
  render();
  dialog.showModal();
}
