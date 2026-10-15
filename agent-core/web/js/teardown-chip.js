/**
 * teardown-chip.js — 「停止智能控制」之后那条设备收尾进度。
 *
 * 停止是两段的，保证不同：
 *
 *   第 1 段  ≤100ms  不会失败     关 agent loop（按钮立刻翻转）
 *   第 2 段  数秒    可能部分失败  逐卡 stop，并行、每卡 5s 死线
 *
 * 这个文件只管第 2 段的显示。它不 import 任何东西、也不碰 `document`：状态机和
 * HTML 都是纯函数，所以 `node --test` 覆盖得到。canvas.js 负责把返回的字符串塞进
 * 元素、并代理「重试」的点击。
 *
 * 为什么不是模态弹窗（启动用的是弹窗）：按下「停止」的那一刻，操作者通常是要去
 * 干别的了 —— 关掉画布、去碰机器人。一个挡住画布、要求他确认的弹窗，在他已经得到
 * 想要的东西（agent 停了）之后才出现，只会被随手点掉，而里面恰好写着「有一台设备
 * 没停下来」。所以它是工具条上一条可以不理、但不会自己消失的提示。
 */

/** 一轮收尾的状态。`items` 按后端 begin 事件的顺序排。 */
export function newTeardown(cards) {
  return {
    items: (cards || []).map((c) => ({
      tool: c.tool || '',
      mcpId: c.mcp_id || '',
      cardId: c.card_id || '',
      status: 'pending',          // pending | stopped | error
      message: '',
    })),
    done: false,
  };
}

/** 收到一条 `project_stop_item`：把对应那项标掉。未知的项忽略。 */
export function applyItem(state, payload) {
  const p = payload || {};
  const item = state.items.find((i) => i.cardId && i.cardId === p.card_id)
    || state.items.find((i) => i.tool === p.tool && i.mcpId === p.mcp_id);
  if (!item) return state;
  item.status = p.status === 'stopped' ? 'stopped' : 'error';
  item.message = p.message || '';
  return state;
}

/** 收到 `project_stop_done`：收尾结束。还卡在 pending 的项当作失败 —— 它们没有
 *  回过话，而「没有回话」在这里的后果和「报错」一样：那台设备可能还在跑。 */
export function applyDone(state) {
  state.items.forEach((i) => {
    if (i.status === 'pending') {
      i.status = 'error';
      i.message = i.message || '没有结果';
    }
  });
  state.done = true;
  return state;
}

export function failedItems(state) {
  return state.items.filter((i) => i.status === 'error');
}

/**
 * 芯片上那一行字。三种形态，说的是**后果**而不是进度百分比：
 *
 *   进行中   ⏳ 设备收尾 4/6
 *   全好     ✓ 设备已收尾
 *   有失败   ⚠ 1 台设备未停止
 *
 * 最后一种刻意不写成「5/6 完成」—— 操作者要知道的不是完成率，而是「还有东西在
 * 跑」。
 */
export function chipLabel(state) {
  const total = state.items.length;
  const settled = state.items.filter((i) => i.status !== 'pending').length;
  const failed = failedItems(state).length;
  if (!state.done) {
    return { icon: '⏳', text: `设备收尾 ${settled}/${total}`, tone: 'busy' };
  }
  if (failed === 0) {
    return { icon: '✓', text: '设备已收尾', tone: 'ok' };
  }
  return { icon: '⚠', text: `${failed} 台设备未停止`, tone: 'warn' };
}

/** 收尾全好时芯片可以自己消失；有失败时**不能** —— 那条信息没人会再去找。 */
export function autoHideMs(state) {
  return state.done && failedItems(state).length === 0 ? 4000 : 0;
}

function _esc(str) {
  return String(str == null ? '' : str).replace(/[&<>"']/g, (ch) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
  ));
}

/**
 * 展开后的逐项列表。失败的那几项排在**最前面**并带「重试」—— 一个 6 张卡片的收尾
 * 里，唯一需要操作的那一项不该要人去扫一遍列表才找到。
 */
export function detailHtml(state) {
  const order = { error: 0, pending: 1, stopped: 2 };
  const rows = [...state.items]
    .sort((a, b) => order[a.status] - order[b.status])
    .map((i) => {
      const icon = i.status === 'stopped' ? '✓' : (i.status === 'error' ? '✗' : '·');
      const retry = i.status === 'error'
        ? `<button class="teardown-retry" data-card-id="${_esc(i.cardId)}"
             data-tool="${_esc(i.tool)}" data-mcp-id="${_esc(i.mcpId)}">重试</button>`
        : '';
      const note = i.status === 'error' && i.message
        ? `<span class="teardown-note">${_esc(i.message)}</span>` : '';
      return `<div class="teardown-row ${i.status}">`
        + `<span class="teardown-row-icon">${icon}</span>`
        + `<span class="teardown-row-tool">${_esc(i.tool)}</span>`
        + `${note}${retry}</div>`;
    });
  return rows.join('');
}
