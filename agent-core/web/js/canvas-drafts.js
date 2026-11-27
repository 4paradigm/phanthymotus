export function canvasDrafts(storage, robotId) {
  const prefix = `canvas-draft:${robotId}:`;
  const legacyKey = `canvas-personal-layout:${robotId}`;

  function list() {
    const drafts = [];
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (!key?.startsWith(prefix)) continue;
      const stored = storage.getItem(key);
      if (!stored) continue;
      const draft = parseDraft(stored);
      if (draft) drafts.push({ id: key.slice(prefix.length), ...draft });
    }
    const legacy = parseDraft(storage.getItem(legacyKey), true);
    if (legacy) drafts.push({ id: '', ...legacy });
    return drafts.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  function save(id, layout, initial = false) {
    const key = prefix + id;
    if (initial && storage.getItem(key) !== null) return;
    storage.setItem(key, JSON.stringify({ updatedAt: Date.now(), layout }));
  }

  return { list, save };
}

function parseDraft(raw, legacy = false) {
  try {
    const value = JSON.parse(raw);
    const draft = legacy ? { updatedAt: 0, layout: value } : value;
    return Array.isArray(draft?.layout?.cards) ? draft : null;
  } catch (error) {
    if (error instanceof SyntaxError) return null;
    throw error;
  }
}

export function openCanvasDrafts({ drafts, activeId, applyLabel, blockedReason, canCopy, onApply, onEdit, onCopy }) {
  const dialog = document.createElement('dialog');
  dialog.className = 'canvas-drafts-dialog';
  dialog.setAttribute('aria-labelledby', 'canvas-drafts-heading');

  const header = document.createElement('header');
  const heading = document.createElement('h2');
  heading.id = 'canvas-drafts-heading';
  heading.textContent = '我的草稿';
  const close = action('关闭', () => dialog.close());
  close.autofocus = true;
  header.append(heading, close);

  const explanation = document.createElement('p');
  explanation.textContent = '草稿可独立编辑，应用时才会写入共享画布。';
  dialog.append(header, explanation);

  const items = document.createElement('div');
  items.className = 'canvas-drafts-list';
  for (const { id, updatedAt, layout } of drafts) {
    const row = document.createElement('article');
    const title = document.createElement('h3');
    const date = updatedAt ? new Date(updatedAt).toLocaleString() : '之前保存的卡片';
    title.textContent = id === activeId ? `本次编辑 · ${date}` : date;
    const cards = document.createElement('p');
    const cardNames = layout.cards.slice(0, 4).map(card => card?.driverName || card?.toolName || '卡片');
    const remaining = layout.cards.length - cardNames.length;
    cards.textContent = cardNames.length
      ? `${cardNames.join('、')}${remaining ? `等 ${layout.cards.length} 张卡片` : ''}`
      : '空画布';
    const button = action(applyLabel, () => { dialog.close(); onApply(layout); });
    button.disabled = !!blockedReason;
    row.append(title, cards);
    if (onEdit) row.append(action('编辑草稿', () => { dialog.close(); onEdit(layout); }));
    row.append(button);
    items.append(row);
  }
  if (!drafts.length) {
    const empty = document.createElement('p');
    empty.textContent = '还没有草稿。可以新建草稿，或编辑共享画布自动保存。';
    items.append(empty);
  }
  dialog.append(items);
  if (blockedReason && drafts.length) {
    const reason = document.createElement('p');
    reason.textContent = blockedReason;
    dialog.append(reason);
  }
  if (onEdit) dialog.append(action('新建草稿', () => { dialog.close(); onEdit(null); }));
  if (canCopy) {
    const copy = action('另存当前为草稿', () => { dialog.close(); onCopy(); });
    dialog.append(copy);
  }
  dialog.addEventListener('close', () => dialog.remove(), { once: true });
  document.body.append(dialog);
  dialog.showModal();
}

function action(label, fn) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  button.addEventListener('click', fn);
  return button;
}
