/** USB pairing UI. Device access and persistence remain owned by the driver. */

export function supportsUsbPairing(mcp) {
  return (mcp?.tools || []).some(tool => typeof tool === 'object' && tool.name === 'usb_pairing');
}

export function decodePairingResponse(body) {
  if (body.code !== 200) throw new Error(body.message || '设备配对失败');
  let result = body.data;
  if (Array.isArray(result)) {
    const text = result.find(item => item.type === 'text')?.text;
    if (!text) throw new Error('驱动未返回配对结果，请检查驱动版本');
    try { result = JSON.parse(text); }
    catch { throw new Error('驱动返回的配对结果无法读取'); }
  }
  if (!result || typeof result !== 'object' || !result.state) {
    throw new Error('驱动未返回有效配对状态');
  }
  if (result.error || result.state === 'error') throw new Error(result.message || result.error);
  return result;
}

export async function callUsbPairing(mcpId, action, sessionId, fetchImpl = globalThis.fetch,
                                    ensureEdit = async () => false) {
  // Recheck on every step: the edit lock may be lost while the dialog is open.
  if (!(await ensureEdit())) throw new Error('需要画布编辑权限才能进行 USB 配对');
  const arguments_ = { action };
  if (sessionId) arguments_.session_id = sessionId;
  const response = await fetchImpl(`/api/mcp/${encodeURIComponent(mcpId)}/call`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tool: 'usb_pairing', arguments: arguments_ }),
  });
  if (!response.ok) throw new Error(`无法连接设备（HTTP ${response.status}）`);
  return decodePairingResponse(await response.json());
}

const STEPS = {
  prepare: { title: '1 / 4 · 准备设备',
    text: '先退出遥操并停止设备动作。拔下手套无线接收器和灵巧手转接器的 USB，保留扩展坞及其他设备。开始配对会停止预览并清除本次标定。',
    button: '已拔下，开始配对', action: 'begin' },
  receiver: { title: '2 / 4 · 连接无线接收器',
    text: '只插入一个手套无线接收器。暂时不要插灵巧手转接器，也不要插拔其他 USB 串口设备。',
    button: '已插入接收器', action: 'receiver' },
  hands: { title: '3 / 4 · 连接灵巧手',
    text: '保持接收器连接，再插入一到两个灵巧手转接器，并确认灵巧手已供电、RS485 已接好。此步骤只读取设备身份，不驱动手指。',
    button: '检测已连接的灵巧手', action: 'hands' },
  review: { title: '4 / 4 · 保存配对',
    text: '保存后立即生效，下次启动自动恢复。请保持当前 USB 插孔位置；换口后重新运行配对向导。保存不会启用遥操。',
    button: '保存并应用', action: 'apply' },
  paired: { title: '配对完成',
    text: '设备配置已保存。现在可以重新开始手套预览并完成标定，再显式启用遥操。接收器换口后请重新配对。',
    button: '完成' },
};

export function pairingStep(state) {
  const step = STEPS[state];
  if (!step) throw new Error('配对状态已变化，请关闭并重新打开向导');
  return step;
}

export async function openUsbPairing(mcpId, { onComplete, ensureEdit = async () => false } = {}) {
  if (!(await ensureEdit())) return;
  const callPairing = (action, sessionId) =>
    callUsbPairing(mcpId, action, sessionId, globalThis.fetch, ensureEdit);
  // One modal per browser: two tabs are additionally isolated by driver sessions.
  if (document.querySelector('.usb-pairing-dialog')) return;
  const previousFocus = document.activeElement;
  const dialog = document.createElement('dialog');
  dialog.className = 'usb-pairing-dialog';
  dialog.setAttribute('aria-labelledby', 'usb-pairing-title');
  dialog.style.cssText = 'width:min(520px,calc(100vw - 40px));box-sizing:border-box;padding:26px;border:1px solid var(--border-color,#64748b);border-radius:16px;background:var(--bg-secondary,#17212e);color:var(--text-primary,#f1f5f9);box-shadow:0 24px 80px #0008;line-height:1.65';
  dialog.innerHTML = `
    <h2 id="usb-pairing-title" style="margin:0 0 8px;font-size:22px">USB 设备配对</h2>
    <p data-heading style="color:var(--text-secondary,#94a3b8);margin:0 0 18px"></p>
    <p data-instruction></p>
    <p data-devices style="font-weight:600"></p>
    <p data-error role="alert" style="color:#fb7185;white-space:pre-wrap"></p>
    <div style="display:flex;justify-content:flex-end;gap:12px;margin-top:24px;flex-wrap:wrap">
      <button type="button" data-cancel style="padding:9px 16px;border-radius:8px;cursor:pointer">取消</button>
      <button type="button" data-next style="padding:9px 16px;border:0;border-radius:8px;background:#3b82f6;color:white;cursor:pointer"></button>
    </div>`;
  let state = 'prepare', sessionId = '', busy = false, cancelFailed = false;
  const next = dialog.querySelector('[data-next]');
  const cancel = dialog.querySelector('[data-cancel]');
  const error = dialog.querySelector('[data-error]');
  function render(result) {
    const step = pairingStep(state);
    dialog.querySelector('[data-heading]').textContent = step.title;
    dialog.querySelector('[data-instruction]').textContent = step.text;
    next.textContent = step.button;
    cancel.hidden = state === 'paired';
    const hands = result?.hands || result?.mapping?.hands;
    dialog.querySelector('[data-devices]').textContent = hands
      ? `已识别：${Object.keys(hands).map(side => side === 'left' ? '左灵巧手' : '右灵巧手').join('、')}` : '';
  }
  function setBusy(value) {
    busy = value;
    next.disabled = value;
    cancel.disabled = value;
    dialog.setAttribute('aria-busy', String(value));
  }
  function close() {
    dialog.close();
    dialog.remove();
    previousFocus?.focus();
  }
  async function cancelPairing() {
    if (busy) return;
    if (cancelFailed) { close(); return; }
    setBusy(true);
    try {
      if (sessionId) await callPairing('cancel', sessionId);
      close();
    } catch (err) {
      error.textContent = `${err.message}。可关闭向导；遗留会话将在闲置 10 分钟后自动释放。`;
      cancel.textContent = '关闭向导';
      cancelFailed = true;
    }
    finally { setBusy(false); }
  }
  cancel.addEventListener('click', cancelPairing);
  dialog.addEventListener('cancel', event => { event.preventDefault(); cancelPairing(); });
  next.addEventListener('click', async () => {
    if (busy) return;
    if (state === 'paired') { close(); return; }
    error.textContent = '';
    setBusy(true);
    try {
      const result = await callPairing(pairingStep(state).action, sessionId);
      // Validate before changing the view; an error never advances the step.
      pairingStep(result.state);
      state = result.state;
      sessionId = result.session_id || sessionId;
      if (state === 'paired') { sessionId = ''; onComplete?.(result); }
      render(result);
    } catch (err) { error.textContent = err.message; }
    finally { setBusy(false); }
  });
  render();
  document.body.appendChild(dialog);
  dialog.showModal();
  next.focus();
}
