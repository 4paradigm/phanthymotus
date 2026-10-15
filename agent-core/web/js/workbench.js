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

export async function openWorkbench(target, spec, ui) {
  if (opened) return;
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
  close.onclick = async () => {
    try {
      if (busy) throw new Error('请求处理中，请等待结果。');
      if ((await invoke(target, {action: 'info'})).busy) throw new Error('请先停止动作，等待完成后再关闭。');
      closed = true; port?.close(); overlay.remove(); opened = false; previousFocus?.focus();
    } catch (error) { alert(error.message); }
  };
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
        let result = await invoke(target, {...args, action});
        if (request.method === 'save') {
          if (!result.skill) throw new Error('扩展未返回技能定义');
          const definition = result.skill.attachment ? {...result.skill,
            origin: {mcp_id: target.mcp, instance_id: target.instanceId}} : result.skill;
          result = {...result, installed: await api('/api/skills/import', definition),
            context: await conversationContext(target).catch(() => ({conversation_ready: false, tool: target.tool}))};
        }
        reply.postMessage({id: request.id, result});
      } catch (error) { reply.postMessage({id: request.id, error: error.message}); }
      finally { if (mutation) busy = false; }
    };
    frame.contentWindow.postMessage({protocol: VERSION}, '*', [channel.port2]);
  };
  document.body.append(overlay); close.focus();
}
