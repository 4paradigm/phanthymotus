/** Read-only recording results and files for a rosbag card. */
const MAX_BLOB_BYTES = 128 * 1024 * 1024;

function humanSize(bytes) {
  if (!Number.isFinite(bytes)) return '未知';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let n = bytes, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${i ? n.toFixed(1) : n} ${units[i]}`;
}

function element(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}

async function info(target, view, bagId, offset) {
  const args = {action: 'info', view};
  if (bagId) args.bag_id = bagId;
  if (offset !== undefined) args.offset = offset;
  const res = await fetch(`/api/mcp/${encodeURIComponent(target.mcpId)}/call?timeout_s=10`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tool: target.toolName, arguments: args}),
  });
  const json = await res.json();
  if (!res.ok || json.code !== 200) throw new Error(json.message || json.detail || '读取录制信息失败');
  const data = Array.isArray(json.data) ? json.data.find(x => x.type === 'text')?.text : json.data;
  const payload = typeof data === 'string' ? JSON.parse(data) : data?.result ?? data;
  if (!payload || typeof payload !== 'object') throw new Error('录制服务返回了无效信息');
  if (view !== 'status' && payload.state === 'error') throw new Error(payload.message || '读取失败');
  return payload;
}

async function downloadFile(target, bagId, file) {
  // Ask for the destination while still handling the click: browsers require
  // a user gesture. This permits large recordings without buffering them.
  let handle;
  if (window.showSaveFilePicker) {
    handle = await window.showSaveFilePicker({suggestedName: file.name});
  } else if (file.size_bytes > MAX_BLOB_BYTES) {
    throw new Error('此浏览器无法流式保存大文件；请使用支持文件保存的浏览器，或从机器人主机取回文件。');
  }
  const query = new URLSearchParams({bag: bagId, file: file.name});
  const response = await fetch(`/api/mcp/${encodeURIComponent(target.mcpId)}/rosbag/file?${query}`);
  if (!response.ok) {
    let message = '下载失败';
    try { message = (await response.json()).detail || message; } catch { /* keep message */ }
    throw new Error(message);
  }
  if (handle) {
    if (!response.body) throw new Error('浏览器不支持流式下载');
    const destination = await handle.createWritable();
    await response.body.pipeTo(destination);
    return;
  }
  const lengthHeader = response.headers.get('Content-Length');
  const actualSize = lengthHeader === null ? NaN : Number(lengthHeader);
  if (!Number.isFinite(actualSize) || actualSize < 0 || actualSize > MAX_BLOB_BYTES) {
    await response.body?.cancel();
    throw new Error('文件过大或大小未知，请使用支持流式保存的浏览器，或从机器人主机取回文件。');
  }
  const url = URL.createObjectURL(await response.blob());
  const link = element('a');
  link.href = url; link.download = file.name;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

export function openRosbagManager(target) {
  const overlay = element('div', undefined, 'modal-overlay');
  const modal = element('section', undefined, 'modal');
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('aria-label', '录制状态与文件');
  modal.style.cssText = 'width:720px;max-width:94vw;max-height:90vh;overflow:auto;padding:20px';
  const header = element('div', undefined, 'modal-header');
  header.append(element('strong', '录制状态与文件', 'modal-title'));
  const close = element('button', '✕', 'btn-icon');
  close.setAttribute('aria-label', '关闭');
  header.append(close);
  const status = element('div', '正在读取录制状态…');
  const refresh = element('button', '刷新列表');
  const previous = element('button', '上一页');
  const next = element('button', '下一页');
  const pages = element('span');
  const paging = element('div');
  paging.style.cssText = 'display:flex;gap:10px;align-items:center;margin-top:8px';
  paging.append(previous, pages, next);
  const select = element('select');
  select.setAttribute('aria-label', '选择录制');
  select.style.cssText = 'width:100%;margin:10px 0';
  const detail = element('div');
  const error = element('p');
  error.setAttribute('role', 'status');
  const note = element('p', '取回时请将同一录制的 metadata.yaml、数据文件和录制记录保存到同一文件夹。');
  modal.append(header, status, refresh, paging, select, detail, note, error);
  overlay.append(modal); document.body.append(overlay);
  const previousFocus = document.activeElement;
  close.focus();
  let timer, closed = false, statusBusy = false, selected = '', lastState, offset = 0;
  const finish = () => {
    closed = true; clearInterval(timer); document.removeEventListener('keydown', onKey);
    overlay.remove(); previousFocus?.focus();
  };
  const onKey = event => { if (event.key === 'Escape') finish(); };
  close.onclick = finish;
  document.addEventListener('keydown', onKey);

  async function refreshStatus() {
    if (statusBusy || closed) return;
    statusBusy = true;
    try {
      const data = await info(target, 'status');
      if (closed) return;
      const names = {idle: '空闲', recording: '录制中', stopping: '正在保存', completed: '已结束', error: '失败'};
      const reasons = {requested: '手动停止', duration_limit: '达到时长上限', disk_space_limit: '磁盘剩余空间不足',
        disk_check_failed: '磁盘检查失败', unexpected_exit: '录制进程意外退出', shutdown: '服务关闭'};
      const elapsed = Number.isFinite(data.duration_sec) && data.path ? ` · ${data.duration_sec.toFixed(1)} 秒` : '';
      const reason = data.stop_reason ? ` · ${reasons[data.stop_reason] || data.stop_reason}` : '';
      const stillRunning = data.process_active === true;
      const unconfirmed = data.state === 'recording' ? ' · 尚未确认消息已写入' : '';
      status.textContent = `状态：${names[data.state] || data.state || '未知'}${elapsed}${reason}${unconfirmed}${data.state === 'error' && stillRunning ? ' · 进程尚未退出' : ''}${data.state === 'error' && data.message ? ` · ${data.message}` : ''}`;
      const wasActive = lastState === true;
      const isActive = stillRunning || data.state === 'recording' || data.state === 'stopping';
      lastState = isActive;
      if (wasActive && !isActive) refreshList();
    } catch (e) { if (!closed) status.textContent = `状态暂不可用：${e.message}`; }
    finally { statusBusy = false; }
  }

  async function showBag(id) {
    selected = id; detail.replaceChildren(); error.textContent = '';
    if (!id) return;
    try {
      const bag = await info(target, 'inspect', id);
      if (closed || id !== selected) return;
      if (bag.start_time_sec != null) detail.append(element('p', `开始时间：${new Date(bag.start_time_sec * 1000).toLocaleString()}`));
      detail.append(element('p', `文件总大小：${humanSize(bag.size_bytes)} · 消息数：${bag.message_count ?? '未知'} · 时长：${bag.duration_sec == null ? '未知' : `${bag.duration_sec.toFixed(2)} 秒`}`));
      detail.append(element('p', `容器内位置：${bag.path}`));
      const topics = element('ul');
      for (const topic of bag.topics || []) topics.append(element('li', `${topic.name} · ${topic.type || '未知类型'} · ${topic.message_count} 条`));
      if (!topics.childNodes.length) topics.append(element('li', '没有可确认的话题信息'));
      detail.append(topics);
      for (const warning of bag.warnings || []) detail.append(element('p', warning));
      for (const file of bag.files || []) {
        const row = element('div');
        row.style.cssText = 'display:flex;gap:12px;align-items:center;margin:8px 0';
        const label = element('span', `${file.name} (${humanSize(file.size_bytes)})`);
        label.style.cssText = 'flex:1;overflow-wrap:anywhere';
        const button = element('button', '下载');
        button.disabled = !!bag.recording;
        button.onclick = async () => {
          error.textContent = '';
          try {
            button.disabled = true;
            await downloadFile(target, id, file);
            error.textContent = `已保存：${file.name}`;
          } catch (e) {
            if (e.name !== 'AbortError') error.textContent = e.message;
          } finally { button.disabled = false; }
        };
        row.append(label, button); detail.append(row);
      }
    } catch (e) { if (!closed && id === selected) error.textContent = e.message; }
  }

  async function refreshList() {
    refresh.disabled = true; previous.disabled = true; next.disabled = true; error.textContent = '';
    try {
      const data = await info(target, 'list', null, offset);
      if (closed) return;
      const records = data.recordings || [];
      previous.disabled = offset === 0;
      next.disabled = !data.has_more;
      pages.textContent = `第 ${Math.floor(offset / 100) + 1} 页，共 ${data.total} 条`;
      select.replaceChildren();
      if (!records.length) select.append(element('option', '暂无录制文件'));
      for (const record of records) {
        const option = element('option', `${record.id} · ${new Date(record.modified_at * 1000).toLocaleString()}${record.recording ? ' · 录制中' : ''}`);
        option.value = record.id; select.append(option);
      }
      if (records.some(x => x.id === selected)) select.value = selected;
      await showBag(records.length ? select.value : '');
    } catch (e) {
      if (!closed) { error.textContent = e.message; previous.disabled = offset === 0; }
    }
    finally { refresh.disabled = false; }
  }
  select.onchange = () => showBag(select.value);
  refresh.onclick = refreshList;
  previous.onclick = () => { offset = Math.max(0, offset - 100); refreshList(); };
  next.onclick = () => { offset += 100; refreshList(); };
  refreshStatus(); refreshList();
  timer = setInterval(refreshStatus, 5000);
}
