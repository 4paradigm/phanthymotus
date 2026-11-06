// Named maps and places. All coordinates are in the selected segment's map frame.
import { SlamPreviewRenderer } from './renderers/mapping.js';
import { decodeSlamPreview } from './renderers/slam-preview-codec.js';

function element(tag, text) {
  const el = document.createElement(tag);
  if (text != null) el.textContent = text;
  return el;
}

function button(parent, label, handler, status) {
  const btn = element('button', label);
  btn.type = 'button';
  btn.onclick = async event => {
    event.stopPropagation();
    btn.disabled = true;
    try { await handler(); }
    catch (error) { status.textContent = error.message; }
    finally { btn.disabled = false; }
  };
  parent.append(btn);
  return btn;
}

async function openLibrary(call, ensureEdit, initialId) {
  const dialog = element('dialog');
  dialog.className = 'slam-library';
  dialog.innerHTML = `<header><strong>已保存地图</strong><span class="slam-view-tools"></span>
    <button type="button" aria-label="关闭">×</button></header>
    <section class="slam-columns"><aside></aside><main>
      <div class="slam-segments"></div><div class="slam-viewport" hidden></div>
      <div class="slam-places"></div></main></section><footer role="status"></footer>`;
  document.body.append(dialog);
  const list = dialog.querySelector('aside'), status = dialog.querySelector('footer');
  const heading = dialog.querySelector('strong'), toolbar = dialog.querySelector('.slam-view-tools');
  const viewport = dialog.querySelector('.slam-viewport'), places = dialog.querySelector('.slam-places');
  const segments = dialog.querySelector('.slam-segments');
  let map, view, socket, retry, output, hasFrame = false, tags = [], generation = 0, busy = false;
  function clearStream() {
    clearTimeout(retry);
    if (socket) { socket.onclose = null; socket.onmessage = null; socket.close(); socket = null; }
    view?.unmount();
    view = null;
    viewport.replaceChildren();
    places.replaceChildren();
  }
  async function closeView() {
    generation++;
    clearStream();
    const previous = map;
    map = output = null;
    tags = [];
    segments.replaceChildren();
    viewport.hidden = true;
    heading.textContent = '已保存地图';
    if (previous) await call('close_map_view', {map_id: previous.id});
  }
  dialog.querySelector('header > button').onclick = () => dialog.close();
  dialog.onclose = () => {
    closeView().catch(error => console.warn('[slam-library]', error.message));
    dialog.remove();
  };
  dialog.showModal();

  function renderTags() {
    places.replaceChildren();
    if (!hasFrame) return;
    const visible = tags.filter(t => t.segment_id === output.segment_id);
    view.setTags(visible);
    button(places, '添加地点', async () => {
      if (!await ensureEdit()) return;
      const current = output;
      status.textContent = '在二维地图点击地点，或拖动设置朝向';
      view.setPickHandler(async point => {
        view.setPickHandler(null);
        const name = window.prompt('地点名称')?.trim();
        if (!name || current !== output) return;
        const selected = map;
        try {
          await call('tag_place', {map_id: selected.id, revision: selected.revision,
            segment_id: current.segment_id, tag_name: name, ...point});
          await refreshTags(selected);
          if (selected === map) status.textContent = '地点已保存';
        } catch (error) { status.textContent = error.message; }
      });
    }, status);
    for (const tag of visible) {
      const row = element('span');
      row.append(element('span', tag.name));
      const remove = button(row, '×', async () => {
        if (!await ensureEdit() || !window.confirm('删除地点「' + tag.name + '」？')) return;
        const selected = map;
        await call('untag_place', {map_id: selected.id, revision: selected.revision, tag_id: tag.id});
        await refreshTags(selected);
      }, status);
      remove.setAttribute('aria-label', '删除地点 ' + tag.name);
      places.append(row);
    }
  }
  async function refreshTags(selected) {
    const result = await call('list_tags', {map_id: selected.id, revision: selected.revision});
    if (selected === map && view) { tags = result.tags; renderTags(); }
  }
  function showSegment(selected) {
    clearStream();
    output = selected;
    hasFrame = false;
    viewport.hidden = false;
    view = Object.create(SlamPreviewRenderer);
    view.mount(viewport, {toolbar, scope: 'history'});
    view.setTagSelectHandler(tag => { status.textContent = tag.name; });
    status.textContent = '等待读取地图点云…';
    const connect = () => {
      const url = new URL('/ws/bus' + selected.topic, location.href);
      url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      socket = new WebSocket(url);
      socket.binaryType = 'arraybuffer';
      socket.onmessage = event => {
        if (event.data instanceof ArrayBuffer) {
          const packet = decodeSlamPreview(event.data);
          if (packet?.meta.segment_id !== selected.segment_id) return;
          view.onData(event.data);
          if (!hasFrame) {
            hasFrame = true;
            renderTags();
            status.textContent = '';
          }
        } else {
          try {
            const message = JSON.parse(event.data);
            if (message.type === 'error') status.textContent = message.message;
          } catch {}
        }
      };
      socket.onclose = () => {
        status.textContent = '地图数据流已断开，正在重连…';
        retry = setTimeout(connect, 2000);
      };
    };
    connect();
  }
  async function select(entry) {
    if (busy) return;
    busy = true;
    try {
      await closeView();
      if (!dialog.open) return;
      const ticket = generation;
      status.textContent = '正在打开地图…';
      const result = await call('select_map', {map_id: entry.id}, true);
      if (ticket !== generation) { await call('close_map_view', {map_id: entry.id}); return; }
      map = {...entry, revision: result.revision};
      heading.textContent = entry.name + ' · v' + result.revision;
      const outputs = result.topic_out.filter(item => item.format === 'sensor/slam-preview');
      if (!outputs.length) throw Error('此地图没有可显示的子图');
      if (outputs.length > 1) {
        const picker = element('select');
        picker.setAttribute('aria-label', '子图');
        for (const [i, item] of outputs.entries()) picker.add(new Option(item.segment_id, String(i)));
        picker.onchange = () => showSegment(outputs[Number(picker.value)]);
        segments.append(picker);
      }
      showSegment(outputs[0]);
      await refreshTags(map);
    } finally { busy = false; }
  }
  async function refresh() {
    const {maps} = await call('list_maps');
    if (!dialog.open) return;
    list.replaceChildren();
    if (!maps.length) list.append(element('p', '暂无已保存地图'));
    for (const entry of maps) {
      const row = element('div');
      button(row, entry.name, () => select(entry), status);
      button(row, '删除', async () => {
        if (busy || !await ensureEdit()) return;
        if (!window.confirm('删除「' + entry.name + '」的 ' + entry.revision +
            ' 个版本及 ' + entry.tag_count + ' 个地点？')) return;
        busy = true;
        try {
          if (map?.id === entry.id) await closeView();
          await call('delete_map', {map_id: entry.id});
          await refresh();
          status.textContent = '地图已删除';
        } finally { busy = false; }
      }, status);
      list.append(row);
    }
    if (initialId) {
      const initial = maps.find(entry => entry.id === initialId);
      initialId = null;
      if (initial) await select(initial);
    }
  }
  try { await refresh(); } catch (error) { status.textContent = error.message; }
}

export function mountSlamActions(container, mcpId, instanceId, ensureEdit) {
  const status = element('output');
  status.setAttribute('role', 'status');
  async function call(action, args = {}, background = false) {
    async function request(arguments_) {
      const response = await fetch('/api/mcp/' + encodeURIComponent(mcpId) + '/call', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({tool: 'pointcloud_slam', arguments: {instance_id: instanceId, ...arguments_}}),
      });
      const json = await response.json();
      if (json.code !== 200) throw Error(json.message || '地图操作失败');
      const result = Array.isArray(json.data) ? JSON.parse(json.data[0].text) : json.data;
      if (!result || result.state === 'error' || result.error) throw Error(result?.message || result?.error || '地图操作失败');
      return result;
    }
    const requestId = background ? Array.from(crypto.getRandomValues(new Uint32Array(4)), n => n.toString(16).padStart(8,'0')).join('') : null;
    let result = await request({action, ...args, ...(background ? {background, request_id: requestId} : {})});
    while (background && result.state === 'working') {
      await new Promise(resolve => setTimeout(resolve, 500));
      result = await request({action: 'operation_status', request_id: requestId});
    }
    return result;
  }
  button(container, '开始建图', async () => {
    if (!await ensureEdit()) return;
    const name = window.prompt('新地图名称')?.trim();
    if (!name) return;
    await call('start_mapping', {map_name: name});
    status.textContent = '正在建图 · ' + name;
  }, status);
  button(container, '停止并保存', async () => {
    if (!await ensureEdit()) return;
    status.textContent = '正在保存…';
    const result = await call('stop_mapping', {}, true);
    showStop(result);
  }, status);
  button(container, '已保存地图', () => openLibrary(call, ensureEdit), status);
  function showStop(result) {
    if (!result.map_id) {
      status.textContent = result.saved === false ? '没有可保存的关键帧' : '没有待保存的地图';
      return;
    }
    const count = Number.isInteger(result.keyframes) ? ' · ' + result.keyframes + ' 个关键帧' : '';
    status.textContent = '已保存 · ' + result.map_name + count + ' ';
    button(status, '查看地图', () => openLibrary(call, ensureEdit, result.map_id), status);
  }
  async function refreshStatus() {
    const info = await call('info');
    if (info.map_name) {
      status.textContent = (info.state === 'save_pending' ? '待重试保存 · ' : '正在建图 · ') + info.map_name;
    } else if (info.last_stop) showStop(info.last_stop);
  }
  container.addEventListener('slam-stop-result', () => refreshStatus().catch(error => {
    status.textContent = error.message;
  }));
  refreshStatus().catch(() => {}); // The card can render before its service connects.
  container.append(status);
}
