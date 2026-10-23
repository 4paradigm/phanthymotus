// Providers declare a read-only listing action; file bytes use the shared volume.
export function fileDownloadSpec(schema) {
  const spec = schema?.['x-file-download'];
  return spec?.protocol === 'motus.files/1' && typeof spec.action === 'string' && spec.action.trim()
    ? { protocol: spec.protocol, action: spec.action } : null;
}

export async function listDownloadFiles(source, mediaType = 'all') {
  const spec = fileDownloadSpec({ 'x-file-download': source.fileDownload });
  if (!spec) throw new Error('文件来源未声明支持的下载能力');
  const args = { action: spec.action };
  if (source.multiInstance) args.instance_id = source.id;
  const response = await fetch(`/api/mcp/${encodeURIComponent(source.mcpId)}/call?timeout_s=20`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tool: source.toolName, arguments: args }),
  });
  if (!response.ok) throw new Error(`无法读取文件列表（HTTP ${response.status}）`);
  const result = await response.json();
  if (result.code !== 200) throw new Error(result.message || '无法读取文件列表');
  const payload = Array.isArray(result.data)
    ? result.data.find(part => part.type === 'text')?.text : result.data;
  const data = typeof payload === 'string' ? JSON.parse(payload) : payload;
  if (data?.ok !== true || !Array.isArray(data.files)) throw new Error(data?.message || '文件来源返回了无效列表');
  return data.files.filter(file => file && typeof file.name === 'string' && file.name.trim() &&
    !/[\\/]/.test(file.name) && typeof file.path === 'string' && file.path.startsWith('/') &&
    file.path.split('/').pop() === file.name && Number.isFinite(file.size) && file.size > 0)
    .map(file => ({ ...file, mime_type: typeof file.mime_type === 'string' ? file.mime_type : 'application/octet-stream' }))
    .filter(file => mediaType === 'photo' ? file.mime_type.startsWith('image/')
      : mediaType === 'video' ? file.mime_type.startsWith('video/') : true);
}

export async function downloadFile(path) {
  // fetch is patched by auth.js; a plain link would bypass its Bearer header.
  const response = await fetch('/api/file/download?path=' + encodeURIComponent(path));
  if (!response.ok) throw new Error(`下载失败（HTTP ${response.status}），文件可能已删除或访问权限已失效`);
  const blob = await response.blob();
  if (!blob.size) throw new Error('文件为空，无法下载');
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = path.split('/').pop();
  try {
    document.body.append(link);
    link.click();
  } finally {
    link.remove();
    // Allow the browser to consume the Blob before releasing it.
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }
}

export function openFileDownloads(getSources, mediaType) {
  const previousFocus = document.activeElement;
  const title = mediaType === 'photo' ? '下载照片' : mediaType === 'video' ? '下载视频' : '文件下载';
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.innerHTML = `<section class="modal file-downloads" role="dialog" aria-modal="true">
    <header class="modal-header"><strong class="modal-title"></strong>
      <div><button type="button" class="btn-ghost" data-refresh>刷新</button>
      <button type="button" class="btn-icon" aria-label="关闭">✕</button></div></header>
    <div class="modal-body"><p role="status" aria-live="polite"></p><ul></ul></div></section>`;
  overlay.querySelector('.modal-title').textContent = title;
  overlay.querySelector('section').setAttribute('aria-label', title);
  const status = overlay.querySelector('[role="status"]');
  const list = overlay.querySelector('ul');
  const refresh = overlay.querySelector('[data-refresh]');
  const close = overlay.querySelector('.btn-icon');
  const dismiss = () => {
    document.removeEventListener('keydown', onKeydown, true);
    overlay.remove();
    previousFocus?.focus();
  };
  function onKeydown(event) {
    if (event.key === 'Escape') { event.stopPropagation(); dismiss(); }
    if (event.key === 'Tab') {
      const buttons = [...overlay.querySelectorAll('button:not(:disabled)')];
      const first = buttons[0], last = buttons.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  }
  async function reload() {
    refresh.disabled = true;
    list.replaceChildren();
    status.textContent = '正在读取保存目录…';
    try {
      const sources = getSources();
      if (!sources.length) {
        status.textContent = '请先将支持文件下载的输出口连接到下载卡片。';
        return;
      }
      const results = await Promise.allSettled(sources.map(source => listDownloadFiles(source, mediaType)));
      if (!overlay.isConnected) return;
      const connected = new Set(getSources().map(source => source.id));
      const files = [], errors = [];
      results.forEach((result, i) => {
        const source = sources[i];
        if (!connected.has(source.id)) return;
        const label = source.driverName || source.mcpId;
        if (result.status === 'fulfilled') files.push(...result.value.map(file => ({ ...file, source, label })));
        else errors.push(`${label}：${result.reason.message}`);
      });
      status.textContent = files.length ? `共 ${files.length} 个文件，点击下载保存到本机。`
        : '暂无可下载文件，可在生成完成后刷新。';
      if (errors.length) status.textContent += ' ' + errors.join('；');
      for (const file of files) {
        const row = document.createElement('li');
        const name = document.createElement('span');
        name.textContent = `${file.name} · ${file.label}`;
        name.title = file.name;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn-ghost';
        button.textContent = '下载';
        button.setAttribute('aria-label', `下载 ${file.name}`);
        button.onclick = async () => {
          button.disabled = true;
          try {
            if (!getSources().some(source => source.id === file.source.id)) throw new Error('文件来源已断开，请刷新列表');
            await downloadFile(file.path);
            status.textContent = `已发起下载：${file.name}`;
          } catch (error) { status.textContent = error.message; }
          finally { button.disabled = false; }
        };
        row.append(name, button);
        list.append(row);
      }
    } catch (error) { status.textContent = error.message; }
    finally { refresh.disabled = false; }
  }
  close.onclick = dismiss;
  refresh.onclick = reload;
  overlay.onclick = event => { if (event.target === overlay) dismiss(); };
  document.addEventListener('keydown', onKeydown, true);
  document.body.append(overlay);
  close.focus();
  reload();
}
