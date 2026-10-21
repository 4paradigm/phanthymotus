// Downloads use Agent Core's existing authenticated file APIs and shared volume.
// vision_capture is the first provider for the independent file_download card.
async function captureInfo(mcpId) {
  const response = await fetch(`/api/mcp/${encodeURIComponent(mcpId)}/call?timeout_s=20`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tool: 'vision_capture', arguments: { action: 'info' } }),
  });
  if (!response.ok) throw new Error(`无法读取拍摄信息（HTTP ${response.status}）`);
  const result = await response.json();
  if (result.code !== 200) throw new Error(result.message || '无法读取拍摄信息');
  const data = Array.isArray(result.data)
    ? result.data.find(part => part.type === 'text')?.text : result.data;
  return typeof data === 'string' ? JSON.parse(data) : data;
}

export async function listCaptureFiles(mcpId, mediaType) {
  const info = await captureInfo(mcpId);
  const dir = mediaType === 'photo' ? info?.photos_dir : info?.videos_dir;
  if (typeof dir !== 'string' || !dir.startsWith('/')) throw new Error('driver 未返回有效的保存目录');
  const path = dir.replace(/\/+$/, '');
  const response = await fetch('/api/file/list', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });
  if (!response.ok) throw new Error(`无法读取保存目录（HTTP ${response.status}），请确认目录已创建且已共享给 Agent Core`);
  const result = await response.json();
  if (result.code !== 200 || !Array.isArray(result.data?.files)) throw new Error(result.message || '无法读取保存目录');
  // Inspect AFTER listing: a recording may have started while the list was read.
  const activePath = mediaType === 'video' ? (await captureInfo(mcpId))?.active_recording?.path : null;
  const extension = mediaType === 'photo' ? /\.(jpe?g|png)$/i : /\.mp4$/i;
  return result.data.files.filter(file => file.type === 'file' && file.size > 0 &&
    typeof file.name === 'string' && !/[\\/]/.test(file.name) && !file.name.startsWith('.') &&
    extension.test(file.name) && `${path}/${file.name}` !== activePath)
    .sort((a, b) => b.name.localeCompare(a.name))
    .map(file => ({ name: file.name, path: `${path}/${file.name}`, size: file.size }));
}

export async function downloadCaptureFile(path) {
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

export function openCaptureDownloads(getSources, mediaType) {
  const previousFocus = document.activeElement;
  const title = mediaType === 'photo' ? '下载照片' : '下载视频';
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.innerHTML = `<section class="modal vision-capture-downloads" role="dialog" aria-modal="true">
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
        status.textContent = '请先将 vision_capture 的文件输出口连接到下载卡片。';
        return;
      }
      const results = await Promise.allSettled(sources.map(source => listCaptureFiles(source.mcpId, mediaType)));
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
        : (mediaType === 'photo' ? '暂无已保存的照片。' : '暂无已完成的视频，录制结束后可刷新。');
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
            await downloadCaptureFile(file.path);
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
