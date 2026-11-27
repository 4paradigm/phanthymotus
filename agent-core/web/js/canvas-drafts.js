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
