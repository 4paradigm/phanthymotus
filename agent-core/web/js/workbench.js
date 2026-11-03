/** A card result may provide an isolated interactive page. The driver owns its
 * contents; the host grants only the selected card's advertised capabilities.
 * No global entry point, robot logic, credentials, or custom stylesheet here.
 */
let opened = false;
const VERSION = 'motus.workbench/1';

async function api(url, body) {
  const response = await fetch(url, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok || result.code !== 200) throw new Error(result.error || result.message || result.detail || '请求失败');
  return result.data;
}

async function conversationContext(target) {
  const layout = await api('/api/canvas/layout');
  const coreIds = new Set((layout.cards || []).filter(c => c.mcpId === 'agentcore').map(c => c.id));
  return {conversation_ready: (layout.execConnections || []).some(c => coreIds.has(c.fromCardId) &&
    c.toMcpId === target.mcp && c.toToolName === target.tool), tool: target.tool};
}

async function connectedTools(target) {
  const layout = await api('/api/canvas/layout');
  const cards = layout.cards || [], connections = layout.execConnections || [];
  const coreIds = new Set(cards.filter(c => c.mcpId === 'agentcore').map(c => c.id));
  const owners = new Set(connections.filter(c => coreIds.has(c.fromCardId) &&
    c.toMcpId === target.mcp && c.toToolName === target.tool &&
    (!target.instanceId || c.toCardId === target.instanceId)).map(c => c.fromCardId));
  const linked = connections.filter(c => owners.has(c.fromCardId));
  const results = await Promise.all([...new Set(linked.map(c => c.toMcpId))].map(async mcp => {
    const data = await api(`/api/mcp/${encodeURIComponent(mcp)}/tools`);
    return (Array.isArray(data) ? data : data.tools || []).flatMap(t =>
      linked.filter(c => c.toMcpId === mcp && c.toToolName === t.name).map(c => ({
        name: `mcp__${mcp}__${t.name}`, description: t.description,
        inputSchema: t.inputSchema, instance_id: c.toCardId,
      })));
  }));
  return {tools: results.flat(), conversation_ready: owners.size > 0};
}

async function invoke(target, args) {
  const result = await api(`/api/mcp/${encodeURIComponent(target.mcp)}/call?timeout_s=20`, {
    tool: target.tool, arguments: {...args, instance_id: target.instanceId || ''},
  });
  const data = Array.isArray(result) ? result.find(c => c.type === 'text')?.text : result?.result ?? result;
  const parsed = typeof data === 'string' ? JSON.parse(data) : data;
  if (!parsed || typeof parsed !== 'object') throw new Error('工作台返回了无效结果');
  if (parsed.error) throw new Error(parsed.error);
  if (Array.isArray(result) && result.some(c => c.type === 'image'))
    return {...parsed, images: result.filter(c => c.type === 'image')};
  return parsed;
}

/**
 * 不经过画布，直接打开一个工具的运维页。
 *
 * ModelHub 这类常驻服务不参与画布的起停，放上画布只是为了能点开它的运维页 ——
 * 侧边栏和决策核心的「打开模型管理」走这里，先调它的入口动作拿到页面再打开。
 */
export async function openToolConsole(mcpId, tool, onClose) {
  const spec = tool?.inputSchema?.['x-workbench'];
  if (spec?.protocol !== VERSION) throw new Error('这个工具没有运维页');
  const target = {mcp: mcpId, tool: tool.name, instanceId: ''};
  const ui = await invoke(target, {action: spec.entry});
  await openWorkbench(target, spec, ui, onClose);
}

/** 本机 ModelHub 的模型管理页：找到在线的那个、有运维页的工具，打开它。 */
export async function openModelConsole(onClose) {
  const mcps = await api('/api/mcp');
  for (const m of mcps || []) {
    if (m.category !== 'modelhub' || m.online === false) continue;
    const tool = (m.tools || []).find(t => typeof t === 'object' && t.inputSchema?.['x-workbench']);
    if (tool) return openToolConsole(m.id, tool, onClose);
  }
  throw new Error('这台机器没有在线的 ModelHub');
}

// Reuse Canvas's shared configuration store. The iframe may configure only
// its own explicitly advertised fields; saved secrets never cross back into it.
async function configureTool(target, spec, args) {
  if (!Array.isArray(spec.config) || !spec.config.length || !spec.actions.includes('config'))
    throw new Error('工作台未声明配置能力');
  if (!args || typeof args !== 'object' || Array.isArray(args) ||
      Object.keys(args).some(k => k !== 'values')) throw new Error('无效配置请求');
  const all = await api(`/api/mcp/${encodeURIComponent(target.mcp)}/tools`);
  const tool = (Array.isArray(all) ? all : all.tools || []).find(t => t.name === target.tool);
  const props = tool?.configSchema?.properties || {};
  const fields = spec.config.filter(k => props[k] && props[k].scope !== 'instance');
  const path = `/api/canvas/tool-config/${encodeURIComponent(target.mcp)}/${encodeURIComponent(target.tool)}`;
  const saved = await api(path) || {};
  if ('values' in args) {
    const values = args.values;
    if (!values || typeof values !== 'object' || Array.isArray(values) ||
        Object.entries(values).some(([k, v]) => !fields.includes(k) ||
          props[k].type !== 'string' || typeof v !== 'string' ||
          (props[k].enum && !props[k].enum.includes(v)))) throw new Error('配置字段或值无效');
    const {ensureEdit, isProjectRunning} = await import('./canvas.js');
    if (isProjectRunning()) throw new Error('请先停止项目，再修改模型设置');
    if (!(await ensureEdit())) throw new Error('请先取得画布编辑权');
    if ((await invoke(target, {action: 'info'})).busy) throw new Error('请等待当前分析完成，再修改模型设置');
    // Validate/apply before saving. Retain omitted keys (including API keys).
    const merged = {...saved, ...values};
    await invoke(target, {action: 'config', ...Object.fromEntries(fields.filter(k => k in merged).map(k => [k, merged[k]]))});
    const response = await fetch(path, {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(merged)});
    const result = await response.json();
    if (!response.ok || result.code !== 200) throw new Error('设置已应用，但保存失败；请重试保存');
    const {getToolConfigs} = await import('./sidebar.js');
    getToolConfigs()[`${target.mcp}:${target.tool}`] = merged;
    return {saved: true};
  }
  const state = await invoke(target, {action: 'info'}), values = {}, secret_set = {}, runtime_secret_set = {};
  for (const k of fields) {
    if (props[k]['x-sensitive'] || props[k].format === 'password') {
      secret_set[k] = !!saved[k];
      runtime_secret_set[k] = state.config_secrets?.[k] === true;
    } else {
      values[k] = saved[k] ?? state.config_values?.[k] ?? props[k].default ?? '';
    }
  }
  return {values, secret_set, runtime_secret_set};
}

async function readSkills(args) {
  if (!args || typeof args !== 'object' || Array.isArray(args) ||
      Object.keys(args).some(k => !['slug', 'cursor', 'limit', 'query'].includes(k)))
    throw new Error('无效的只读技能查询');
  const summary = s => ({slug: String(s.slug || '').slice(0, 100), name: String(s.name || '').slice(0, 200),
    oneLiner: String(s.oneLiner || '').slice(0, 500), description: String(s.description || '').slice(0, 600),
    category: String(s.category || '').slice(0, 80), version: String(s.version || '').slice(0, 80),
    active: s.active === true, requiredTools: Array.isArray(s.requiredTools) ? s.requiredTools.filter(v => typeof v === 'string').slice(0, 32).map(v => v.slice(0, 200)) : []});
  if ('slug' in args) {
    if (Object.keys(args).length !== 1 || typeof args.slug !== 'string' || !/^[A-Za-z0-9_.-]{1,100}$/.test(args.slug))
      throw new Error('无效技能标识');
    const s = await api(`/api/skills/${encodeURIComponent(args.slug)}`);
    if (typeof s.instruction !== 'string' || s.instruction.length > 32000)
      throw new Error('技能指令过大，请在原技能面板查看');
    return {skill: {...summary(s), instruction: s.instruction}};
  }
  const cursor = args.cursor ?? 0, limit = args.limit ?? 30, query = args.query ?? '';
  if (!Number.isInteger(cursor) || cursor < 0 || !Number.isInteger(limit) || limit < 1 || limit > 30 ||
      typeof query !== 'string' || query.length > 200) throw new Error('技能查询范围无效');
  const all = await api('/api/skills');
  const rows = (Array.isArray(all) ? all : []).map(summary).filter(s =>
    !query || `${s.name} ${s.oneLiner} ${s.description}`.toLowerCase().includes(query.toLowerCase()));
  return {skills: rows.slice(cursor, cursor + limit), next_cursor: cursor + limit < rows.length ? cursor + limit : null};
}

export async function openWorkbench(target, spec, ui, onClose) {
  if (opened) throw new Error('请先关闭已打开的工作台再继续。草稿已保留。');
  if (spec.protocol !== VERSION || !Array.isArray(spec.actions) ||
      !spec.actions.every(a => typeof a === 'string') || !spec.actions.includes('info') ||
      !spec.actions.includes('stop') || ui.protocol !== VERSION ||
      typeof ui.html !== 'string' || ui.html.length > 250000) throw new Error('不支持的工作台页面');
  opened = true;
  const previousFocus = document.activeElement;
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.dataset.workbench = '';
  overlay.innerHTML = `<section class="modal" role="dialog" aria-modal="true" style="width:960px;height:720px;max-height:94dvh">
    <header class="modal-header"><strong class="modal-title"></strong><button class="btn-icon" aria-label="关闭工作台">✕</button></header>
    <iframe sandbox="allow-scripts" referrerpolicy="no-referrer" style="border:0;width:100%;flex:1;min-height:0"></iframe></section>`;
  const title = spec.title || target.tool;
  overlay.querySelector('.modal-title').textContent = title;
  overlay.querySelector('section').setAttribute('aria-label', title);
  // aria-label, not title: Safari shows an iframe's title as a tooltip wherever the pointer
  // rests inside the frame, so a workbench page had "本地大模型" floating over its own UI.
  const frame = overlay.querySelector('iframe'); frame.setAttribute('aria-label', title);
  // A later meta cannot relax this policy; the opaque-origin page has no token.
  frame.srcdoc = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; ${spec.media === true ? 'media-src blob:;' : ''} connect-src 'none'; form-action 'none'; base-uri 'none'">${ui.html}`;
  let port, busy = false, closed = false;
  const inputs = () => {
    if (spec.inputs !== true || typeof target.getInputs !== 'function') throw new Error('工作台未声明当前卡片来源');
    return target.getInputs().filter(t => typeof t.topic === 'string' && /^\/[A-Za-z0-9_/]+$/.test(t.topic))
      .map(t => ({topic: t.topic, format: String(t.format || '')}));
  };
  const close = overlay.querySelector('button');

  // Teardown is unconditional once we decide to go. `opened` is module-global,
  // so leaving it true on a failed close locked the page out of every workbench
  // for the rest of the session — and the only way out was a reload, because
  // close was the sole place that cleared it and it threw into an `alert` the
  // moment the device was unreachable.
  const teardown = () => {
    closed = true;
    port?.close();
    document.removeEventListener('keydown', onKeydown, true);
    overlay.remove();
    opened = false;
    previousFocus?.focus();
    onClose?.();
  };

  /** Ask the device first; offer to leave anyway if it cannot answer. */
  const requestClose = async () => {
    if (closed) return;
    if (busy) { alert('请求处理中，请等待结果。'); return; }
    let running;
    try {
      running = (await invoke(target, {action: 'info'})).busy === true;
    } catch (error) {
      // Unreachable is not the same as busy. Say which one it is and let the
      // operator decide, rather than trapping them in the dialog.
      if (!confirm(`无法读取设备状态（${error.message}）。仍要关闭工作台吗？草稿已保留。`)) return;
      teardown();
      return;
    }
    if (running) { alert('请先停止当前操作，等待完成后再关闭。'); return; }
    teardown();
  };

  function onKeydown(event) {
    if (event.key === 'Escape' && !closed) { event.stopPropagation(); requestClose(); }
  }
  document.addEventListener('keydown', onKeydown, true);
  close.onclick = requestClose;
  frame.onload = () => {
    if (closed) return;
    port?.close(); // Navigation revokes the previous document's capability.
    const channel = new MessageChannel(); port = channel.port1;
    port.onmessage = async event => {
      const reply = channel.port1, request = event.data;
      if (closed || !request || typeof request.id !== 'string') return;
      let mutation = false;
      try {
        if (JSON.stringify(request).length > 40000) throw new Error('工作台请求过大');
        if (request.method === 'upload') {
          const file = request.file;
          const limit = Math.min(Number(spec.files?.max_bytes) || 0, 64 * 1024 * 1024);
          if (!(file instanceof Blob) || !limit || file.size > limit || !file.size ||
              !spec.files?.accept?.includes(file.type)) throw new Error('文件类型或大小超出工作台声明');
          if (busy) throw new Error('上一项操作尚未完成');
          mutation = busy = true;
          const form = new FormData(); form.append('file', file, `workbench-${crypto.randomUUID()}`);
          const response = await fetch(`/api/mcp/${encodeURIComponent(target.mcp)}/file/upload`, {method: 'POST', body: form});
          const result = await response.json();
          if (!response.ok || result.code !== 200) throw new Error(result.message || '上传失败');
          reply.postMessage({id: request.id, result: result.data}); return;
        }
        if (request.method === 'media') {
          if (spec.media !== true || !/^[A-Za-z0-9_-]{1,128}$/.test(request.args?.asset_id || '')) throw new Error('无效素材请求');
          const response = await fetch(`/api/mcp/${encodeURIComponent(target.mcp)}/workbench/media`, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({tool: target.tool, instance_id: target.instanceId || '',
              asset_id: request.args.asset_id, kind: 'video'}),
          });
          if (!response.ok) throw new Error((await response.json()).detail || '视频读取失败');
          if (response.headers.get('Content-Type')?.split(';')[0] !== 'video/mp4' ||
              Number(response.headers.get('Content-Length')) > 64 * 1024 * 1024) throw new Error('无效视频响应');
          reply.postMessage({id: request.id, result: {blob: await response.blob()}}); return;
        }
        if (request.method === 'tools') {
          if (spec.tools !== true) throw new Error('工作台未声明工具查询能力');
          reply.postMessage({id: request.id, result: await connectedTools(target)}); return;
        }
        if (request.method === 'skills') {
          if (spec.skills?.read !== true) throw new Error('工作台未声明只读技能查询能力');
          reply.postMessage({id: request.id, result: await readSkills(request.args || {})}); return;
        }
        if (request.method === 'context') {
          reply.postMessage({id: request.id, result: await conversationContext(target)}); return;
        }
        if (request.method === 'inputs') {
          reply.postMessage({id: request.id, result: {inputs: inputs()}}); return;
        }
        if (request.method === 'config') {
          if (busy) throw new Error('上一项操作尚未完成');
          mutation = true; busy = true;
          reply.postMessage({id: request.id, result: await configureTool(target, spec, request.args || {})}); return;
        }
        if (!['call', 'save'].includes(request.method)) throw new Error('不支持的工作台请求');
        const args = request.args;
        if (!args || typeof args !== 'object' || Array.isArray(args)) throw new Error('无效参数');
        const action = request.method === 'save' ? 'save' : args.action;
        if (!spec.actions.includes(action)) throw new Error('此工作台未声明该操作');
        if (request.method === 'call' && action === 'save') throw new Error('请使用技能保存接口');
        if (request.method === 'save' && spec.kind !== 'skill-authoring') throw new Error('此工作台未声明技能保存能力');
        const isMutation = !['info', 'check', 'inspect', 'stop', 'model_catalog',
          'skill_list', 'skill_inspect', 'task_list', 'task_inspect', 'task_check'].includes(action);
        if (isMutation && busy) throw new Error('上一项操作尚未完成');
        mutation = isMutation;
        if (mutation) busy = true;
        const driverArgs = {...args, action};
        if (spec.inputs === true && action === 'start') {
          const {ensureEdit} = await import('./canvas.js');
          if (!(await ensureEdit())) throw new Error('请先取得画布编辑权');
          // The host supplies this card's resolved inputs, never arbitrary iframe topics.
          driverArgs.input_topics = [...new Set(inputs().map(t => t.topic))];
          delete driverArgs.input_topic;
          if (!driverArgs.input_topics.length) throw new Error('请先在画布把相机连到此卡片');
        }
        const autoExecute = driverArgs.auto_execute;
        if (autoExecute !== undefined) {
          if (request.method !== 'save' || typeof autoExecute !== 'boolean' || !spec.actions.includes('execution_binding'))
            throw new Error('此工作台不支持自动执行设置');
          delete driverArgs.auto_execute;
        }
        let result = await invoke(target, driverArgs);
        if (request.method === 'save') {
          if (!result.skill) throw new Error('扩展未返回技能定义');
          const definition = result.skill.attachment ? {...result.skill,
            origin: {mcp_id: target.mcp, instance_id: target.instanceId}} : result.skill;
          result = {...result, installed: await api('/api/skills/import', definition),
            context: await conversationContext(target).catch(() => ({conversation_ready: false, tool: target.tool}))};
          if (autoExecute !== undefined) {
            result.execution_policy = await api(`/api/skills/package/execution/${encodeURIComponent(definition.slug)}`, {enabled: autoExecute});
          }
        }
        reply.postMessage({id: request.id, result});
      } catch (error) { reply.postMessage({id: request.id, error: error.message}); }
      finally { if (mutation) busy = false; }
    };
    frame.contentWindow.postMessage({protocol: VERSION}, '*', [channel.port2]);
  };
  try {
    document.body.append(overlay);
    close.focus();
  } catch (error) {
    // Nothing is mounted yet, so release the latch rather than wedging it.
    document.removeEventListener('keydown', onKeydown, true);
    overlay.remove();
    opened = false;
    throw error;
  }
}
