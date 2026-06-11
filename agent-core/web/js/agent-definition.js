/**
 * agent-definition.js — 智能体定义 modal（分 tab 编辑）。
 * Tabs: 身份定义 / 系统提示词 / 长期记忆
 */

const overlay    = document.getElementById('agent-def-overlay');
const btnOpen    = document.getElementById('btn-agent-def');
const btnClose   = document.getElementById('agent-def-close');
const btnCancel  = document.getElementById('agent-def-cancel');
const btnSave    = document.getElementById('agent-def-save');
const taIdentity = document.getElementById('agent-def-identity');
const taSystem   = document.getElementById('agent-def-system');
const taMemory   = document.getElementById('agent-def-memory');
const tabs       = document.querySelectorAll('.agent-def-tab');
const panels     = document.querySelectorAll('.agent-def-panel');

// Tab switching
tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.tab;
    tabs.forEach(t => t.classList.toggle('active', t === tab));
    panels.forEach(p => p.classList.toggle('active', p.dataset.panel === target));
  });
});

function show() {
  overlay.classList.remove('hidden');
  // Reset to first tab
  tabs[0].click();
  // Load content
  fetch('/api/agent/definition')
    .then(r => r.json())
    .then(res => {
      if (res.code === 200 && res.data) {
        taIdentity.value = res.data.identity || '';
        taSystem.value = res.data.system || '';
        taMemory.value = res.data.memory || '';
      }
    })
    .catch(err => console.error('[agent-def] load failed:', err));
}

function hide() {
  overlay.classList.add('hidden');
}

async function save() {
  btnSave.disabled = true;
  btnSave.textContent = '保存中…';
  try {
    const resp = await fetch('/api/agent/definition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        identity: taIdentity.value,
        system: taSystem.value,
        memory: taMemory.value,
      }),
    });
    const res = await resp.json();
    if (res.code === 200) {
      btnSave.textContent = '已保存';
      setTimeout(() => { btnSave.textContent = '保存'; }, 1500);
    } else {
      alert('保存失败: ' + (res.message || '未知错误'));
      btnSave.textContent = '保存';
    }
  } catch (err) {
    alert('保存失败: ' + err.message);
    btnSave.textContent = '保存';
  } finally {
    btnSave.disabled = false;
  }
}

btnOpen.addEventListener('click', show);
btnClose.addEventListener('click', hide);
btnCancel.addEventListener('click', hide);
btnSave.addEventListener('click', save);

export { show, hide };
