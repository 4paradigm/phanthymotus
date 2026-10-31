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

async function invoke(target, args) {
  const result = await api(`/api/mcp/${encodeURIComponent(target.mcp)}/call?timeout_s=20`, {
    tool: target.tool, arguments: {...args, instance_id: target.instanceId || ''},
  });
  const data = Array.isArray(result) ? result.find(c => c.type === 'text')?.text : result?.result ?? result;
  const parsed = typeof data === 'string' ? JSON.parse(data) : data;
  if (!parsed || typeof parsed !== 'object') throw new Error('工作台返回了无效结果');
  if (parsed.error) throw new Error(parsed.error);
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
  const frame = overlay.querySelector('iframe'); frame.title = title;
  // A later meta cannot relax this policy; the opaque-origin page has no token.
  frame.srcdoc = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; form-action 'none'; base-uri 'none'">${ui.html}`;
  let port, busy = false, closed = false;
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
        if (request.method === 'context') {
          reply.postMessage({id: request.id, result: await conversationContext(target)}); return;
        }
        if (!['call', 'save'].includes(request.method)) throw new Error('不支持的工作台请求');
        const args = request.args;
        if (!args || typeof args !== 'object' || Array.isArray(args)) throw new Error('无效参数');
        const action = request.method === 'save' ? 'save' : args.action;
        if (!spec.actions.includes(action)) throw new Error('此工作台未声明该操作');
        if (request.method === 'call' && action === 'save') throw new Error('请使用技能保存接口');
        if (request.method === 'save' && spec.kind !== 'skill-authoring') throw new Error('此工作台未声明技能保存能力');
        const isMutation = !['info', 'check', 'inspect', 'stop'].includes(action);
        if (isMutation && busy) throw new Error('上一项操作尚未完成');
        mutation = isMutation;
        if (mutation) busy = true;
        const driverArgs = {...args, action};
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
